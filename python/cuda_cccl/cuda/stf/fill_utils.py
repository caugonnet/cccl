# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Fill / init for STF logical data (cuda.core.Buffer.fill; CuPy/Numba fallback for 8-byte).
"""

import numpy as np

from cuda.core import Buffer, Stream


def init_logical_data(ctx, ld, value, data_place=None, exec_place=None):
    """
    Initialize a logical data with a constant value.

    Uses cuda.core.Buffer.fill for 1/2/4-byte element types. For 8-byte types
    (e.g. float64, int64), falls back to CuPy if available, else a Numba kernel.

    Parameters
    ----------
    ctx : context
        STF context
    ld : logical_data
        Logical data to initialize
    value : scalar
        Value to fill the array with
    data_place : data_place, optional
        Data place for the initialization task
    exec_place : exec_place, optional
        Execution place for the fill operation
    """
    # Create write dependency with optional data place
    dep_arg = ld.write(data_place) if data_place else ld.write()

    # Create task arguments - include exec_place if provided
    task_args = []
    if exec_place is not None:
        task_args.append(exec_place)
    task_args.append(dep_arg)

    with ctx.task(*task_args) as t:
        # Logical data index: 1 if exec_place was passed, else 0
        ld_index = 1 if exec_place is not None else 0
        cai = t.get_arg_cai(ld_index)
        ptr = cai["data"][0]
        shape = tuple(cai["shape"])
        dtype = np.dtype(cai["typestr"])
        size = int(np.prod(shape)) * dtype.itemsize

        core_stream = Stream.from_handle(t.stream_ptr())
        buf = Buffer.from_handle(ptr, size, owner=None)

        if dtype.itemsize in (1, 2, 4):
            # cuda.core.Buffer.fill supports int [0,256) or 1/2/4-byte pattern
            if value == 0 or value == 0.0:
                fill_val = 0
            else:
                fill_val = np.array([value], dtype=dtype).tobytes()
            buf.fill(fill_val, stream=core_stream)
        else:
            # 8-byte or other: CuPy if available, else Numba kernel
            _fill_large_element(shape, dtype, value, ptr, size, t.stream_ptr())


def _fill_large_element(shape, dtype, value, ptr, size, stream_ptr):
    """Fill buffer when element size is not 1/2/4 bytes (e.g. float64)."""
    try:
        import cupy as cp

        mem = cp.cuda.UnownedMemory(ptr, size, owner=None)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        arr = cp.ndarray(shape, dtype=dtype, memptr=memptr)
        with cp.cuda.ExternalStream(stream_ptr):
            arr.fill(value)
    except ImportError:
        # Fallback: Numba kernel when CuPy unavailable
        from numba import cuda

        nb_stream = cuda.external_stream(stream_ptr)
        array = cuda.from_cuda_array_interface(
            {
                "data": (ptr, False),
                "shape": shape,
                "typestr": dtype.str,
                "version": 2,
            },
            owner=None,
            sync=False,
        )
        _fill_with_simple_kernel(array, value, nb_stream)


def _make_fill_kernels():
    """Build Numba JIT kernels only when needed (lazy)."""
    from numba import cuda

    @cuda.jit
    def _fill_kernel_fallback(array, value):
        idx = cuda.grid(1)
        if idx < array.size:
            array.flat[idx] = value

    @cuda.jit
    def _zero_kernel_fallback(array):
        idx = cuda.grid(1)
        if idx < array.size:
            array.flat[idx] = 0

    return _fill_kernel_fallback, _zero_kernel_fallback


_fill_kernel_fallback = None
_zero_kernel_fallback = None


def _get_fill_kernels():
    global _fill_kernel_fallback, _zero_kernel_fallback
    if _fill_kernel_fallback is None:
        _fill_kernel_fallback, _zero_kernel_fallback = _make_fill_kernels()
    return _fill_kernel_fallback, _zero_kernel_fallback


def _fill_with_simple_kernel(array, value, stream):
    """Fallback using a Numba JIT kernel (8-byte types when CuPy unavailable)."""
    fill_kernel, zero_kernel = _get_fill_kernels()
    total_size = array.size
    threads_per_block = 256
    blocks_per_grid = (total_size + threads_per_block - 1) // threads_per_block

    if value == 0 or value == 0.0:
        zero_kernel[blocks_per_grid, threads_per_block, stream](array)
    else:
        typed_value = array.dtype.type(value)
        fill_kernel[blocks_per_grid, threads_per_block, stream](array, typed_value)
