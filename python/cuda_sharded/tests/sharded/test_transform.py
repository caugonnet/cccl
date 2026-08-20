# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tier-2 first rung: transform with standard-op descriptors."""

import numpy as np
import pytest

pytest.importorskip("cuda.sharded._experimental._sharded_bindings")

import cuda.sharded._experimental as shd


def _pair(group, h):
    a = shd.sharded_array.from_numpy(group, h)
    out = shd.sharded_array.allocate(group, h.shape[0], h.dtype)
    return a, out


class TestUnary:
    def test_negate(self, group):
        h = np.arange(-50, 50, dtype=np.float32)
        a, out = _pair(group, h)
        shd.transform(group, a, out, "negate")
        np.testing.assert_array_equal(out.to_numpy(), -h)

    def test_negate_int(self, group3):
        h = np.arange(10, dtype=np.int32)
        a, out = _pair(group3, h)
        shd.transform(group3, a, out, "negate")
        np.testing.assert_array_equal(out.to_numpy(), -h)

    def test_scale(self, group):
        h = np.arange(1000, dtype=np.float64)
        a, out = _pair(group, h)
        shd.transform(group, a, out, "scale", alpha=2.5)
        np.testing.assert_allclose(out.to_numpy(), 2.5 * h)

    def test_scale_int64(self, group):
        h = np.arange(1000, dtype=np.int64)
        a, out = _pair(group, h)
        shd.transform(group, a, out, "scale", alpha=3)
        np.testing.assert_array_equal(out.to_numpy(), 3 * h)

    def test_add_scalar(self, group):
        h = np.arange(500, dtype=np.float32)
        a, out = _pair(group, h)
        shd.transform(group, a, out, "add_scalar", alpha=10.5)
        np.testing.assert_allclose(out.to_numpy(), h + np.float32(10.5))

    def test_in_place(self, group):
        h = np.arange(100, dtype=np.float32)
        a = shd.sharded_array.from_numpy(group, h)
        shd.transform(group, a, a, "scale", alpha=2.0)
        np.testing.assert_array_equal(a.to_numpy(), 2 * h)

    def test_unknown_op(self, group):
        h = np.arange(10, dtype=np.float32)
        a, out = _pair(group, h)
        with pytest.raises(ValueError):
            shd.transform(group, a, out, "sqrt")


class TestBinary:
    def test_add(self, group):
        x = np.arange(1000, dtype=np.float32)
        y = np.full(1000, 10.0, dtype=np.float32)
        a = shd.sharded_array.from_numpy(group, x)
        b = shd.sharded_array.from_numpy(group, y)
        out = shd.sharded_array.allocate(group, 1000, np.float32)
        shd.transform_binary(group, a, b, out, "add")
        np.testing.assert_array_equal(out.to_numpy(), x + y)

    def test_mul(self, group3):
        x = np.arange(1, 11, dtype=np.int64)
        y = np.arange(11, 21, dtype=np.int64)
        a = shd.sharded_array.from_numpy(group3, x)
        b = shd.sharded_array.from_numpy(group3, y)
        out = shd.sharded_array.allocate(group3, 10, np.int64)
        shd.transform_binary(group3, a, b, out, "mul")
        np.testing.assert_array_equal(out.to_numpy(), x * y)

    def test_axpy(self, group):
        x = np.arange(1000, dtype=np.float64)
        y = np.ones(1000, dtype=np.float64)
        a = shd.sharded_array.from_numpy(group, x)
        b = shd.sharded_array.from_numpy(group, y)
        out = shd.sharded_array.allocate(group, 1000, np.float64)
        shd.transform_binary(group, a, b, out, "axpy", alpha=0.5)
        np.testing.assert_allclose(out.to_numpy(), 0.5 * x + y)

    def test_shape_mismatch(self, group):
        a = shd.sharded_array.allocate(group, 100, np.float32)
        b = shd.sharded_array.allocate(group, 101, np.float32)
        out = shd.sharded_array.allocate(group, 100, np.float32)
        with pytest.raises(ValueError):
            shd.transform_binary(group, a, b, out, "add")

    def test_dtype_mismatch(self, group):
        a = shd.sharded_array.allocate(group, 100, np.float32)
        b = shd.sharded_array.allocate(group, 100, np.float64)
        out = shd.sharded_array.allocate(group, 100, np.float32)
        with pytest.raises(ValueError):
            shd.transform_binary(group, a, b, out, "add")
