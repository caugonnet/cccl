# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Per-shard __cuda_array_interface__ interop: external kernels write, sharded
algorithms read (and the other way around); adoption of external buffers."""

import numpy as np
import pytest

pytest.importorskip("cuda.sharded._experimental._sharded_bindings")
cp = pytest.importorskip("cupy")

import cuda.sharded._experimental as shd  # noqa: E402  (must follow importorskip)


class TestShardViewsWithCupy:
    def test_per_shard_writes_algorithm_reads(self, group):
        """cupy writes each shard through the CAI view; a whole-array numpy
        copy and a sharded reduce both see the writes."""
        n = 10_000
        a = shd.sharded_array.allocate(group, n, np.float32)
        for v in a.shards():
            cv = cp.asarray(v)
            assert cv.shape == (v.size,)
            cv[:] = cp.float32(v.index + 1)
        cp.cuda.runtime.deviceSynchronize()

        h = a.to_numpy()
        expected = np.concatenate(
            [np.full(v.size, v.index + 1, np.float32) for v in a.shards()]
        )
        np.testing.assert_array_equal(h, expected)
        ref = float(expected.astype(np.float64).sum())
        assert abs(shd.reduce(group, a, op="sum") - ref) < 1e-3 * abs(ref)

    def test_contiguous_per_shard_writes_whole_array_read(self, group):
        """Per-shard cupy writes into a CONTIGUOUS array read back through a
        whole-array numpy copy (one VA range, placed pages)."""
        n = 1 << 16
        a = shd.sharded_array.allocate_contiguous(group, n, np.int32)
        for v in a.shards():
            cv = cp.asarray(v)
            cv[:] = cp.arange(v.global_offset, v.global_offset + v.size, dtype=cp.int32)
        cp.cuda.runtime.deviceSynchronize()
        np.testing.assert_array_equal(a.to_numpy(), np.arange(n, dtype=np.int32))

    def test_algorithm_writes_cupy_reads(self, group):
        a = shd.sharded_array.allocate(group, 1000, np.float64)
        shd.iota(group, a)
        pieces = [cp.asnumpy(cp.asarray(v)) for v in a.shards()]
        np.testing.assert_array_equal(
            np.concatenate(pieces), np.arange(1000, dtype=np.float64)
        )


class TestAdopt:
    def test_adopt_and_reduce(self, group):
        rng = np.random.default_rng(21)
        sizes = [1000 + 100 * i for i in range(group.size)]
        host = [rng.random(s).astype(np.float64) for s in sizes]
        bufs = [cp.asarray(h) for h in host]
        cp.cuda.runtime.deviceSynchronize()

        a = shd.sharded_array.adopt(group, bufs)
        assert a.size == sum(sizes)
        assert a.num_shards == group.size
        ref = np.concatenate(host)
        np.testing.assert_array_equal(a.to_numpy(), ref)
        np.testing.assert_allclose(
            shd.reduce(group, a, op="sum"), ref.sum(), rtol=1e-12
        )

    def test_adopt_algorithm_writes_visible_in_source(self, group):
        bufs = [cp.zeros(100, dtype=cp.int32) for _ in range(group.size)]
        a = shd.sharded_array.adopt(group, bufs)
        shd.fill(group, a, 9)
        for b in bufs:
            np.testing.assert_array_equal(cp.asnumpy(b), np.full(100, 9, np.int32))

    def test_adopt_dtype_mismatch(self, group):
        if group.size < 2:
            pytest.skip("needs >= 2 places")
        bufs = [cp.zeros(10, dtype=cp.float32), cp.zeros(10, dtype=cp.float64)]
        bufs += [cp.zeros(10, dtype=cp.float32) for _ in range(group.size - 2)]
        with pytest.raises(ValueError):
            shd.sharded_array.adopt(group, bufs)


class TestNumba:
    def test_numba_kernel_per_shard(self, group):
        numba_cuda = pytest.importorskip("numba.cuda")

        @numba_cuda.jit
        def double_it(x):
            i = numba_cuda.grid(1)
            if i < x.shape[0]:
                x[i] *= 2.0

        h = np.arange(4096, dtype=np.float64)
        a = shd.sharded_array.from_numpy(group, h)
        for v in a.shards():
            dev = numba_cuda.as_cuda_array(v)
            double_it[(v.size + 255) // 256, 256](dev)
        numba_cuda.synchronize()
        np.testing.assert_array_equal(a.to_numpy(), 2 * h)
