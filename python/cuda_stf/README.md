# CUDA STF Python Package

[`cuda.stf._experimental`](https://nvidia.github.io/cccl/python/stf.html)
provides Python bindings to **CUDASTF (Sequential Task Flow)**: you define logical
data and submit tasks that read or write that data, and STF infers the
dependencies and orchestrates execution and data movement. It is part of the
[CUDA Core Compute Libraries](https://nvidia.github.io/cccl/cpp.html#cccl-cpp-libraries).

The API is exposed under the `_experimental` subpackage because it is still
evolving and may change without notice. CUDASTF is currently **Linux-only**.

## Installation

Install from PyPI:

```bash
pip install cuda-stf[cu13]  # For CUDA 13.x (pip-installed cuda-toolkit)
pip install cuda-stf[cu12]  # For CUDA 12.x (pip-installed cuda-toolkit)
```

If you already have a CUDA toolkit on your system and do not want pip to
install it, use the `sysctk` variants:

```bash
pip install cuda-stf[sysctk13]  # For CUDA 13.x (system CUDA toolkit)
pip install cuda-stf[sysctk12]  # For CUDA 12.x (system CUDA toolkit)
```

For a smaller install without Numba (when you drive kernels through
`cuda.core` / `cuda.compute` or your own launches), use the `minimal-*`
variants:

```bash
pip install cuda-stf[minimal-cu13]       # pip CUDA toolkit, no Numba
pip install cuda-stf[minimal-sysctk13]   # system CUDA toolkit, no Numba
```

Install `cuda-cccl` as well when using `cuda.compute` with STF or compiling
external C++ code against the cudax headers; it supplies the libcudacxx, CUB,
and Thrust headers.

Feature dependencies are installed separately as needed: `cuda-cccl`
(`cuda.compute` and header discovery), `numba` / `numba-cuda` (Numba interop,
bundled by the non-`minimal` extras), `cupy`, `torch` (PyTorch interop),
`warp-lang` (Warp interop), and `nvmath-python` (cuBLAS/cuSOLVER examples).

### Install from source (Linux only)

```bash
git clone https://github.com/NVIDIA/cccl.git
cd cccl/python/cuda_stf
pip install -e .[test-cu13]  # or .[test-cu12], .[test-sysctk13], .[test-sysctk12]
```

Building from source compiles the native `cccl.c.experimental.stf` / `cudax`
extension, so a C++ toolchain and CMake (`>=3.30`) with Ninja are required in
addition to the CUDA toolkit. The `test-*` extras add `cuda-cccl`, `pytest`,
`pytest-xdist`, and CuPy so the test suite (`pytest tests/`) can run.

**Requirements:** Python 3.10+, CUDA Toolkit 12.x or 13.x, NVIDIA GPU with
Compute Capability 7.5+, Linux.

## Thread safety

The bindings support free-threaded CPython (3.13t/3.14t): the extension
module declares `freethreading_compatible`, so importing `cuda.stf` does not
re-enable the GIL, and the wrapper objects synchronize their own lifetime
management. Free-threaded submitters are useful when task submission is
host-side bound. The contract below applies to GIL and free-threaded builds
alike.

**Safe from multiple threads:**

- **Task submission on a shared context.** Several threads may concurrently
  create `logical_data`, build and run tasks (`ctx.task(...)`,
  `ctx.cuda_kernel(...)`), and let wrapper objects be garbage-collected, all
  against one shared `context`. The underlying runtime synchronizes its
  submission path (per-logical-data locking, mutex-guarded stream pools);
  conflicting accesses to the *same* logical data are ordered through the
  task dependency graph, exactly as in the C++ API.
- **Dropping wrapper references.** A `logical_data`, `task`, or kernel
  wrapper may be released by any thread at any time, including concurrently
  with `finalize()` on another thread: each context carries a lifetime lock
  that orders child destruction against context finalization.
- **`finalize()` itself is idempotent and single-shot.** If several threads
  race to finalize the same context, exactly one performs the teardown and
  the rest are no-ops.
- **`check_errors()`**: each pending host-callback error is surfaced to
  exactly one caller.
- **`with exec_place:` affinity scopes** are per-thread (and nestable):
  each thread that enters a place gets its own scope, exited by the matching
  `__exit__` on that thread.

**Requires external quiescence (single-caller phase operations):**

- `finalize()`, `fence()`, and `wait()` must not run concurrently with
  in-flight task submission on the same context. Stop the submitter threads
  (join them or use a barrier), then call the phase operation. This mirrors
  the C++ contract. After `finalize()`, submission attempts raise
  `RuntimeError`.
- `stackable_context` scope structure (`push()`/`pop()`, `graph_scope`,
  `while_loop`, `repeat`) is thread-confined by design: record, and release
  recorded graphs (`LaunchableGraph.reset()` or the last reference drop), on
  the thread that entered the context. Duplicate concurrent `reset()` calls
  are guarded (at most one release happens), and resetting an already-reset
  graph is a safe no-op from any thread.
- `TaskGraph` recording (`with graph:`) is single-threaded; `launch()` from
  one thread at a time.
- Context configuration (e.g. `exec_place.set_affine_data_place`) must
  happen before threads start submitting through that place.

## Documentation

For complete documentation, examples, and API reference, visit:

- **Full Documentation**: [nvidia.github.io/cccl/python/stf.html](https://nvidia.github.io/cccl/python/stf.html)
- **Repository**: [github.com/NVIDIA/cccl](https://github.com/NVIDIA/cccl)
- **Examples**: [github.com/NVIDIA/cccl/tree/main/python/cuda_stf/tests/stf](https://github.com/NVIDIA/cccl/tree/main/python/cuda_stf/tests/stf)
