# cuda-sharded: Python bindings for sharded containers and algorithms

Experimental Python bindings for the `cuda::experimental::sharded` C++ layer
(cudax): 1-D arrays partitioned into placed shards — one physical home per
element across locality domains or devices — plus algorithms that run each
place's piece locally and combine across places through the shared address
space.

```python
import numpy as np
import cuda.sharded._experimental as shd

group = shd.place_group.by_locality_domains()  # or .by_devices([0, 1])
a = shd.sharded_array.from_numpy(group, np.arange(1 << 20, dtype=np.float32))
shd.sort(group, a)
total = shd.reduce(group, a, op="sum")
```

## Design: bind the containers, tier the algorithms

**Containers are bindings, not a reimplementation.** `place_group`,
`sharded_array` and the fixed-size/contiguity contract live once, in the C++
headers; Python holds opaque handles. Contract violations surface as Python
exceptions (`std::invalid_argument` → `ValueError`, `std::runtime_error` →
`RuntimeError`).

**Algorithms are tiered by what crosses the Python/C++ boundary:**

1. **Opaque algorithms** (`fill`, `sequence`/`iota`, `reduce` with standard
   ops, `inclusive_scan`/`exclusive_scan`, `adjacent_difference`, `count`,
   `histogram_even`, `sort`): one Python → C++ crossing per call; the C++
   side owns the per-shard loop, the per-place streams, and the cross-place
   combine. This is the default path.
2. **Operator-parameterized algorithms** (`transform`, `transform_binary`):
   a descriptor set of standard ops (`negate`, `scale`, `add_scalar`; `add`,
   `mul`, `axpy`) lowers to the pure C++ path — the operator is a small enum,
   not a callback, so there is still exactly one crossing per call.
   *Follow-up (not in this package yet):* arbitrary Python operators compiled
   once via the `cuda.compute` numba → LTO-IR machinery, cached by
   `(op, dtype, algorithm)` and executed by C++ per shard — the operator
   crosses the boundary once as compiled code, never per call.
3. **Per-shard escape hatch** (always available):
   `sharded_array.shard(i)` returns a zero-copy view exposing
   `__cuda_array_interface__` (data pointer, shape, typestr, and the shard's
   reference stream), so numba / cupy / torch kernels can run on individual
   shards directly. The loop this implies is over *places*, not elements —
   fine at millisecond scale, the wrong tool inside tight loops.

Supported dtypes: `float32`, `float64`, `int32`, `int64` (explicitly
instantiated in the C++ shim).

All algorithm bindings are synchronous: they return when the result is ready
on every place.

`sort` note: the engine behind the sharded name is the distributed sort of
the multi-GPU layer. This binding targets correctness; throughput follows the
engine's, and engine improvements land here transparently (the engine slot
is a performance detail, not part of this API). Shard sizes/offsets are preserved by the engine's final
redistribution, so contiguous arrays remain valid after sorting.

## Free-threading (3.13t/3.14t) contract

The extension is built with the Cython directive
`freethreading_compatible=True` and holds **no mutable module-level state**
(the dtype/op tables are created at import and never mutated), so importing
it does not re-enable the GIL on free-threaded CPythons.

Thread contract (consistent with the `cuda-stf` bindings):

- A `place_group` may be shared by any number of threads; stream lookup and
  stream-color assignment are internally synchronized (C++-side mutex/atomic).
- Distinct `sharded_array` objects may be used concurrently from distinct
  threads, including algorithm calls over them that share one `place_group`.
- Concurrent operations on the **same** `sharded_array` are not synchronized;
  interleaving mutating calls on one array is a data race, exactly as it is
  in C++.
- Blocking shim calls release the GIL (`with nogil`), so threaded host
  dispatch also scales on regular GIL builds.

## Building and installing

The extension compiles a CUDA shim TU against the CCCL headers of the
enclosing repository checkout; you need `nvcc` (CUDA 12.x or 13.x) and a host
C++20 compiler. From the repo:

```bash
cd python/cuda_sharded

# with build isolation (pip provides cython/cmake/ninja):
pip install -e .[test]

# or against an environment that already has cython, cmake, ninja and
# scikit-build-core (e.g. for a specific GPU arch and host compiler):
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=103a -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/c++" \
  pip install -e . --no-build-isolation
```

The default `CMAKE_CUDA_ARCHITECTURES` is CCCL's minimum supported
architecture; pass your GPU's architecture (as above) to avoid JIT-on-import.
The extension links `cudart` statically, so the wheel has no libcudart
runtime dependency; the CUDA driver must be present.

Run the tests:

```bash
pytest tests/
```

The container/algorithm tests need one GPU. Interop tests use `cupy` (and
`numba-cuda` where installed); they skip when those packages are missing.
On free-threaded builds, `tests/sharded/test_threading.py` doubles as the
free-threading smoke test (it asserts the GIL stayed disabled).

## Status and follow-ups

This is the first PR of the Python tier. Deliberately deferred:

- the numba → LTO-IR rung for arbitrary Python operators (tier 2b, cached by
  `(op, dtype, algorithm)`);
- `sharded_csr` + spmv/spmm bindings (behind the cuSPARSE build gate, like
  the C++ layer);
- graph/STF capture of per-iteration sharded sequences (the loop-level tier);
- multi-CUDA wheels (`merge_cuda_wheels.py`-style packaging; a source build
  ships exactly one `cuXX/` directory).
