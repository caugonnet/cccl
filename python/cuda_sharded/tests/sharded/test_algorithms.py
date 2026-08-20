# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tier-1 algorithm bindings vs numpy references."""

import numpy as np
import pytest

pytest.importorskip("cuda.sharded._experimental._sharded_bindings")

import cuda.sharded._experimental as shd

DTYPES = [np.float32, np.float64, np.int32, np.int64]


class TestFill:
    def test_fill_float(self, group):
        a = shd.sharded_array.allocate(group, 1000, np.float32)
        shd.fill(group, a, 42.5)
        np.testing.assert_array_equal(a.to_numpy(), np.full(1000, 42.5, np.float32))

    def test_fill_negative_int(self, group):
        a = shd.sharded_array.allocate(group, 777, np.int32)
        shd.fill(group, a, -7)
        np.testing.assert_array_equal(a.to_numpy(), np.full(777, -7, np.int32))

    def test_fill_large_int64(self, group):
        # would not round-trip through a double
        v = (1 << 62) + 12345
        a = shd.sharded_array.allocate(group, 10, np.int64)
        shd.fill(group, a, v)
        np.testing.assert_array_equal(a.to_numpy(), np.full(10, v, np.int64))


class TestSequence:
    def test_iota(self, group):
        a = shd.sharded_array.allocate(group, 1000, np.int64)
        shd.iota(group, a)
        np.testing.assert_array_equal(a.to_numpy(), np.arange(1000, dtype=np.int64))

    def test_iota_start(self, group3):
        a = shd.sharded_array.allocate(group3, 10, np.int32)
        shd.iota(group3, a, start=5)
        np.testing.assert_array_equal(a.to_numpy(), np.arange(5, 15, dtype=np.int32))

    def test_sequence_step(self, group):
        a = shd.sharded_array.allocate(group, 100, np.float64)
        shd.sequence(group, a, start=1.5, step=0.5)
        np.testing.assert_allclose(a.to_numpy(), 1.5 + 0.5 * np.arange(100), rtol=1e-12)


class TestReduce:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_sum(self, group, dtype):
        rng = np.random.default_rng(42)
        h = rng.integers(-100, 100, 10_000).astype(dtype)
        a = shd.sharded_array.from_numpy(group, h)
        res = shd.reduce(group, a, op="sum")
        if np.dtype(dtype).kind == "f":
            np.testing.assert_allclose(res, h.astype(np.float64).sum(), rtol=1e-5)
        else:
            assert res == int(h.astype(np.int64).sum())

    def test_min_max(self, group):
        rng = np.random.default_rng(7)
        h = rng.permutation(np.arange(1234, dtype=np.float32))
        a = shd.sharded_array.from_numpy(group, h)
        assert shd.reduce(group, a, op="min") == 0.0
        assert shd.reduce(group, a, op="max") == 1233.0

    def test_min_max_int64(self, group):
        h = np.array([5, -3, 9, 0, 9, -3] * 100, dtype=np.int64)
        a = shd.sharded_array.from_numpy(group, h)
        assert shd.reduce(group, a, op="min") == -3
        assert shd.reduce(group, a, op="max") == 9

    def test_unknown_op(self, group):
        a = shd.sharded_array.allocate(group, 10, np.float32)
        with pytest.raises(ValueError):
            shd.reduce(group, a, op="prod")

    def test_uneven(self, group3):
        h = np.arange(1, 11, dtype=np.float32)
        a = shd.sharded_array.from_numpy(group3, h)
        assert abs(shd.reduce(group3, a, op="sum") - 55.0) < 0.1


class TestScan:
    def test_inclusive_int(self, group):
        h = np.ones(2000, dtype=np.int32)
        a = shd.sharded_array.from_numpy(group, h)
        shd.inclusive_scan(group, a)
        np.testing.assert_array_equal(a.to_numpy(), np.arange(1, 2001, dtype=np.int32))

    def test_inclusive_float(self, group3):
        rng = np.random.default_rng(3)
        h = rng.random(1000).astype(np.float64)
        a = shd.sharded_array.from_numpy(group3, h)
        shd.inclusive_scan(group3, a)
        np.testing.assert_allclose(a.to_numpy(), np.cumsum(h), rtol=1e-10)

    def test_exclusive_int(self, group):
        h = np.ones(2000, dtype=np.int64)
        a = shd.sharded_array.from_numpy(group, h)
        shd.exclusive_scan(group, a)
        np.testing.assert_array_equal(a.to_numpy(), np.arange(2000, dtype=np.int64))

    def test_exclusive_uneven(self, group3):
        h = np.arange(1, 11, dtype=np.int32)
        a = shd.sharded_array.from_numpy(group3, h)
        shd.exclusive_scan(group3, a)
        ref = np.concatenate([[0], np.cumsum(h)[:-1]]).astype(np.int32)
        np.testing.assert_array_equal(a.to_numpy(), ref)


class TestAdjacentDifference:
    def test_basic(self, group):
        h = np.cumsum(np.arange(1, 101, dtype=np.int32)).astype(np.int32)
        a = shd.sharded_array.from_numpy(group, h)
        out = shd.sharded_array.allocate(group, 100, np.int32)
        shd.adjacent_difference(group, a, out)
        ref = np.concatenate([[h[0]], np.diff(h)]).astype(np.int32)
        np.testing.assert_array_equal(out.to_numpy(), ref)

    def test_uneven(self, group3):
        h = np.cumsum(np.arange(1, 11, dtype=np.int64)).astype(np.int64)
        a = shd.sharded_array.from_numpy(group3, h)
        out = shd.sharded_array.allocate(group3, 10, np.int64)
        shd.adjacent_difference(group3, a, out)
        np.testing.assert_array_equal(out.to_numpy(), np.arange(1, 11, dtype=np.int64))

    def test_shape_mismatch(self, group):
        a = shd.sharded_array.allocate(group, 100, np.float32)
        out = shd.sharded_array.allocate(group, 101, np.float32)
        with pytest.raises(ValueError):
            shd.adjacent_difference(group, a, out)

    def test_dtype_mismatch(self, group):
        a = shd.sharded_array.allocate(group, 100, np.float32)
        out = shd.sharded_array.allocate(group, 100, np.float64)
        with pytest.raises(ValueError):
            shd.adjacent_difference(group, a, out)


class TestCount:
    def test_count(self, group):
        h = np.array([1, 2, 3, 2, 1, 2, 3, 2] * 50, dtype=np.int32)
        a = shd.sharded_array.from_numpy(group, h)
        assert shd.count(group, a, 2) == int((h == 2).sum())

    def test_count_float(self, group3):
        h = np.zeros(1000, dtype=np.float64)
        h[::3] = 1.25
        a = shd.sharded_array.from_numpy(group3, h)
        assert shd.count(group3, a, 1.25) == int((h == 1.25).sum())


class TestHistogramEven:
    @pytest.mark.parametrize("dtype", [np.int32, np.float32])
    def test_vs_numpy(self, group, dtype):
        rng = np.random.default_rng(11)
        h = rng.integers(0, 100, 10_000).astype(dtype)  # strictly inside [0, 100)
        a = shd.sharded_array.from_numpy(group, h)
        res = shd.histogram_even(group, a, 10, 0, 100)
        ref, _ = np.histogram(h, bins=10, range=(0, 100))
        np.testing.assert_array_equal(res.astype(np.int64), ref)

    def test_out_of_range_ignored(self, group):
        h = np.array([-5, 0, 5, 15, 25], dtype=np.int32)
        a = shd.sharded_array.from_numpy(group, h)
        res = shd.histogram_even(group, a, 2, 0, 20)
        np.testing.assert_array_equal(res.astype(np.int64), [2, 1])  # -5, 25 ignored

    def test_bad_args(self, group):
        a = shd.sharded_array.allocate(group, 10, np.float32)
        with pytest.raises(ValueError):
            shd.histogram_even(group, a, 0, 0.0, 1.0)
        with pytest.raises(ValueError):
            shd.histogram_even(group, a, 4, 1.0, 1.0)


class TestSort:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_random(self, group, dtype):
        rng = np.random.default_rng(99)
        if np.dtype(dtype).kind == "f":
            h = rng.random(20_000).astype(dtype)
        else:
            h = rng.integers(-100_000, 100_000, 20_000).astype(dtype)
        a = shd.sharded_array.from_numpy(group, h)
        shd.sort(group, a)
        np.testing.assert_array_equal(a.to_numpy(), np.sort(h))
        assert a.num_shards == group.size  # layout preserved

    @pytest.mark.parametrize(
        "make",
        [
            lambda: np.arange(5000, dtype=np.int32),  # pre-sorted
            lambda: np.arange(5000, dtype=np.int32)[::-1].copy(),  # reverse
            lambda: np.tile(np.array([5, 3, 5, 1, 3], np.int64), 1000),  # dup-heavy
        ],
    )
    def test_distributions(self, group, make):
        h = make()
        a = shd.sharded_array.from_numpy(group, h)
        shd.sort(group, a)
        np.testing.assert_array_equal(a.to_numpy(), np.sort(h))

    def test_contiguous(self, group):
        rng = np.random.default_rng(5)
        h = rng.random(8192).astype(np.float32)
        a = shd.sharded_array.from_numpy(group, h, contiguous=True)
        shd.sort(group, a)
        assert a.is_contiguous  # sort preserves the layout
        np.testing.assert_array_equal(a.to_numpy(), np.sort(h))
