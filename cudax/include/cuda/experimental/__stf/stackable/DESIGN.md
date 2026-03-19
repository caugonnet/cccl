# Stackable Context Design

## Overview

The `stackable_ctx` introduces nestable CUDA task graph contexts on top of the
existing CUDA STF (Stream Task Flow) model. The core idea is that a user creates
a top-level context (backed by a `stream_ctx`) and then "pushes" nested
sub-contexts (backed by `graph_ctx`), each of which captures work into a CUDA
graph. When a nested context is "popped," its graph is either:

- **Instantiated and launched** on a stream (if the parent is a `stream_ctx`), or
- **Embedded as a child graph node** (if the parent is already a `graph_ctx`),
  enabling arbitrarily deep nesting.

This enables composable library APIs where functions can create their own nested
graph scopes without knowledge of the caller's context, iterative algorithms
expressed as native CUDA conditional graph nodes, and graph caching for
amortized instantiation cost.

## File Organization

| File | Role |
|------|------|
| `stackable_ctx_impl.cuh` | `stackable_ctx` class, context tree, `deferred_task_builder` |
| `stackable_ctx.cuh` | `stackable_logical_data<T>`, `stackable_task_dep<T>`, RAII guards |
| `stackable_node_hierarchy.cuh` | `node_hierarchy` tree (parent/child offsets, free-list pool) |
| `stackable_task_dep.cuh` | Type traits (`is_stackable_task_dep`) and conversion helper |
| `conditional_nodes.cuh` | `push_while_config`, condition update/reset kernels (CUDA 12.4+) |

## Context Hierarchy

The context tree is managed by `stackable_ctx::impl` using a `node_hierarchy`
structure that tracks parent/child relationships via integer offsets. Each node
in the tree corresponds to a `ctx_node_base`:

```
stackable_ctx::impl
├── node_hierarchy (tree of offsets: parent[], children[], free_list)
├── nodes[] (sparse vector of ctx_node_base, indexed by offset)
├── head_map (thread::id -> current offset)
└── async_handles (pooled per exec_place)
```

There are two concrete node types:

- **`stream_ctx_node`**: Used only at the root. Wraps a `stream_ctx`. Its
  `finalize()` is blocking.
- **`graph_ctx_node`**: Used for all nested levels. Wraps a `graph_ctx` that
  captures work into a `cudaGraph_t`. Its `finalize()` either launches the
  graph on a stream (if the parent is a `stream_ctx`) or embeds it as a child
  graph node (if the parent is another `graph_ctx`).

Each thread maintains its own "head" offset (via `head_map`), allowing multiple
threads to work on different parts of the context tree concurrently. Push and
pop operations are serialized with an exclusive lock; task submission acquires a
shared lock.

### Push / Pop Lifecycle

**Push:**
1. Acquire exclusive lock.
2. Allocate a free offset from `node_hierarchy`.
3. Create either a `stream_ctx_node` (root) or `graph_ctx_node` (nested).
4. For graph nodes: create the CUDA graph (child graph node or standalone),
   set up allocator adapters, and configure the graph context.
5. Update the thread's head offset.

**Pop:**
1. Acquire exclusive lock.
2. **Prologue** (`_pop_prologue`): For each pushed data, call
   `pop_before_finalize()` to destroy child logical data and record DOT edges.
3. **Finalize**: Polymorphic dispatch on the context node:
   - `stream_ctx_node`: Synchronous finalize.
   - `graph_ctx_node` (nested): Add graph dependencies, return child node event.
   - `graph_ctx_node` (root-level): Instantiate graph (via cache), launch,
     return stream event.
4. **Epilogue** (`_pop_epilogue`): Unfreeze parent data, recycle the async
   handle, propagate allocator adapters, discard the node, restore head offset.

## Data Model

### Ownership Layers

`stackable_logical_data<T>` uses a two-pointer indirection to manage data
lifetime across context boundaries:

```
stackable_logical_data<T>          (user-facing, copyable via shared_ptr)
└── shared_ptr<impl>               (handle; destructor triggers retain logic)
    └── shared_ptr<state>          (long-lived, type-erased via base class)
        ├── data_nodes[]           (sparse vector indexed by context offset)
        ├── data_root_offset       (where the data was originally created)
        ├── symbol, read_only      (metadata)
        └── mutex                  (per-data locking)
```

**Why two layers?** When the user destroys their `stackable_logical_data`
before calling `pop()`, the `impl` destructor fires and transfers the `state`
(via `shared_ptr`) into child contexts' `retained_data` vectors. This keeps the
data alive until those contexts are popped. If `impl` and `state` were merged
into a single object behind a single `shared_ptr`, there would be no way to
distinguish "user dropped their handle" from "last reference died."

The `state` class inherits from `stackable_logical_data_impl_state_base` to
enable type erasure: context nodes store `vector<shared_ptr<base>>` for pushed
and retained data without knowing `T`.

### Data Nodes

Each `data_node` within `state` represents the logical data at a specific
context level:

```cpp
class data_node {
    logical_data<T> ld;                          // the data at this level
    optional<frozen_logical_data<T>> frozen_ld;  // frozen parent copy (if imported)
    event_list unfreeze_prereqs;                 // events for deferred unfreeze
    int get_cnt;                                 // how many children hold a frozen copy
    access_mode effective_mode;                  // actual access pattern observed
};
```

### Data Movement Between Levels (Push)

When data crosses a context boundary (either explicitly via `data.push()` or
implicitly via automatic push during task submission):

1. **Freeze** the parent's `logical_data` (creates a snapshot via the existing
   STF freeze mechanism).
2. **Get** a copy at the target data place (`frozen_ld.get(where)`).
3. **Create** a new `logical_data` in the child context from that copy.
4. **Track** the push in the child context node's `pushed_data` vector.
5. **Increment** `get_cnt` on the parent's `data_node`.

On pop, the reverse happens: the child's `logical_data` is destroyed, and once
all children have been popped (`get_cnt == 0`), the parent data is unfrozen,
propagating any changes back.

### Automatic Push and Access Mode Validation

When a task references a `stackable_logical_data` that hasn't been explicitly
pushed to the current context level, `validate_access()` triggers an automatic
push. The default push mode is conservative (`rw` for read-write data, `read`
for data marked `set_read_only()`). This ensures correctness at the cost of
serializing accesses across sibling contexts.

If the same logical data appears multiple times in a task's dependencies (e.g.,
`data.read()` and `data.write()`), access modes are combined before pushing.
This is handled by `process_pack()` (for immediate constructs like
`parallel_for`) and by `deferred_task_builder::concretize_deferred_task()` (for
the `task()` builder).

## Task Submission

### Direct Constructs

`parallel_for`, `launch`, `host_launch`, `cuda_kernel`, and `cuda_kernel_chain`
are thin wrappers that:

1. Acquire a shared lock.
2. Call `process_pack()` to validate and auto-push all stackable dependencies.
3. Delegate to the underlying context's method (converting `stackable_task_dep`
   to `task_dep` via `reserved::to_task_dep()`).

### Deferred Task Builder

The `task()` method returns a `deferred_task_builder` that supports incremental
dependency addition via `add_deps()`. Dependencies are not resolved immediately;
instead, the builder stores them and resolves everything at once when the task
is concretized (via `->*` or `.start()`). This two-phase approach ensures that
access modes are combined correctly even when the same data appears in both the
initial dependencies and `add_deps()` calls.

```
ctx.task(data1.read(), data2.write())
   .add_deps(data1.write())    // data1 now needs rw mode
   .set_symbol("my_task")
   ->*[](cudaStream_t s, auto d1, auto d2) { ... };
```

## Conditional Nodes (CUDA 12.4+)

### While Loops

`push_while()` creates a conditional graph node of type `cudaGraphCondTypeWhile`.
The body graph is the child context. A reset kernel is automatically added after
the conditional node to restore the handle for potential re-execution.

The `while_graph_scope_guard` RAII class wraps this pattern. Its `update_cond()`
method provides a fluent API for setting the loop condition from a device lambda:

```cpp
auto while_guard = ctx.while_graph_scope();
while_guard.update_cond(residual.read())->*[tol] __device__(auto residual) {
    return *residual > tol;  // continue while above tolerance
};
```

### Repeat Loops

`repeat_graph_scope_guard` is a higher-level abstraction built on top of while
loops. It automatically manages a counter logical data, initializes it, and sets
up the decrement-and-test condition.

```cpp
{
    auto guard = ctx.repeat_graph_scope(100);
    // body executes 100 times as a single CUDA graph
}
```

## RAII Guards

| Guard | Constructor action | Destructor action |
|-------|-------------------|-------------------|
| `graph_scope_guard` | `push()` | `pop()` |
| `while_graph_scope_guard` | `push_while()` | `pop()` |
| `repeat_graph_scope_guard` | Create counter + `while_graph_scope` + `update_cond` | Destroy while guard |

All guards are non-copyable and non-movable (like `std::lock_guard`). Factory
methods (`graph_scope()`, `while_graph_scope()`, `repeat_graph_scope()`) are
marked `[[nodiscard]]` to prevent accidental immediate destruction.

## Thread Safety

- **`std::shared_mutex`** on `stackable_ctx::impl`: shared for task submission
  and reads, exclusive for push/pop.
- **Per-thread head offset** via `head_map`: each thread tracks its own current
  context position in the tree.
- **Per-data `std::shared_mutex`** on `state`: protects `data_nodes` during
  concurrent validation and push operations.

## Graph Caching

When a non-nested graph context is popped, the CUDA graph is submitted to the
executable graph cache (`async_resources_handle::cached_graphs_query`). The
cache:

1. Looks for an existing executable graph with the same (nnodes, nedges)
   signature.
2. Attempts `cudaGraphExecUpdate` to reuse it (cache hit).
3. On miss, instantiates a new `cudaGraphExec_t` and stores it.
4. Uses LRU eviction when the cache size limit is exceeded.

Statistics (instantiate count, update count per call-site) are collected when
`CUDASTF_DISPLAY_GRAPH_STATS` is set.

## Allocator Adapters

Each graph context node creates a `stream_adapter` that forwards allocations
from the graph capture phase to the parent's stream. This is necessary because
CUDA graph capture cannot allocate memory directly. For nested graph contexts,
adapters are propagated to the parent on pop (rather than cleared immediately)
because the memory must remain valid until the parent graph is launched.

## Known Limitations and Future Work

- **`node_hierarchy` has a fixed initial pool** of 16 offsets with no growth
  mechanism. Deep or wide context trees will assert.
- **Conservative default push mode** (`rw`) serializes sibling contexts. Users
  must call `set_read_only()` to enable concurrent read access.
- **Freeze-based data movement** always copies. There is no zero-copy path for
  data already in the right place.
- **The `impl`/`state` split**, while motivated, creates complex ownership.
  Renaming and removing the duplicate `sctx` stored in both layers would improve
  clarity (see plan: simplify data model layers).
