# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest

import cuda.compute


def test_generate_constant():
    """Generate fills every element with the same constant."""
    num_items = 1024
    d_out = cp.empty(num_items, dtype=np.int32)

    def constant_42():
        return 42

    cuda.compute.generate(d_out, constant_42, num_items)

    expected = np.full(num_items, 42, dtype=np.int32)
    np.testing.assert_array_equal(d_out.get(), expected)


def test_generate_float():
    """Generate works with floating-point output."""
    num_items = 512
    d_out = cp.empty(num_items, dtype=np.float64)

    def pi():
        return 3.14159265

    cuda.compute.generate(d_out, pi, num_items)

    expected = np.full(num_items, 3.14159265, dtype=np.float64)
    np.testing.assert_allclose(d_out.get(), expected)


def test_generate_with_stream(cuda_stream):
    """Generate respects the stream argument."""
    num_items = 256
    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)

    with cp_stream:
        d_out = cp.empty(num_items, dtype=np.int32)

    def constant_7():
        return 7

    cuda.compute.generate(d_out, constant_7, num_items, stream=cuda_stream)

    expected = np.full(num_items, 7, dtype=np.int32)
    np.testing.assert_array_equal(d_out.get(), expected)


def test_generate_object_api():
    """The make_generate object API works and can be called repeatedly."""
    num_items = 128
    d_out = cp.empty(num_items, dtype=np.int32)

    def constant_99():
        return 99

    generator = cuda.compute.make_generate(d_out, constant_99)

    generator(d_out, constant_99, num_items)
    expected = np.full(num_items, 99, dtype=np.int32)
    np.testing.assert_array_equal(d_out.get(), expected)

    # Call again to verify reuse works
    generator(d_out, constant_99, num_items)
    np.testing.assert_array_equal(d_out.get(), expected)


def test_generate_stateful():
    """Generate with a stateful operator referencing a device array."""
    num_items = 64
    d_out = cp.empty(num_items, dtype=np.int32)
    d_value = cp.array([123], dtype=np.int32)

    def read_value():
        return d_value[0]

    cuda.compute.generate(d_out, read_value, num_items)

    expected = np.full(num_items, 123, dtype=np.int32)
    np.testing.assert_array_equal(d_out.get(), expected)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_generate_dtypes(dtype):
    """Generate works across common dtypes."""
    num_items = 256
    d_out = cp.empty(num_items, dtype=dtype)

    def zero():
        return 0

    cuda.compute.generate(d_out, zero, num_items)

    expected = np.zeros(num_items, dtype=dtype)
    np.testing.assert_array_equal(d_out.get(), expected)
