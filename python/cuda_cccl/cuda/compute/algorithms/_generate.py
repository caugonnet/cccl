# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Callable

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_with_registered_key_functions
from .._cccl_interop import set_cccl_iterator_state
from .._utils import protocols
from ..op import OpAdapter, make_op_adapter
from ..typing import DeviceArrayLike, IteratorT, Operator


class _Generate:
    __slots__ = ["d_out_cccl", "op_cccl", "build_result"]

    def __init__(
        self,
        d_out: DeviceArrayLike | IteratorT,
        op: OpAdapter,
    ):
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)

        out_type = cccl.get_value_type(d_out)
        self.op_cccl = op.compile((), out_type)

        self.build_result = cccl.call_build(
            _bindings.DeviceGenerate,
            self.d_out_cccl,
            self.op_cccl,
        )

    def __call__(
        self,
        d_out,
        op: Callable | OpAdapter,
        num_items: int,
        stream=None,
    ):
        op_adapter = make_op_adapter(op)

        set_cccl_iterator_state(self.d_out_cccl, d_out)
        self.op_cccl.state = op_adapter.get_state()

        stream_handle = protocols.validate_and_get_stream(stream)
        self.build_result.compute(
            self.d_out_cccl,
            num_items,
            self.op_cccl,
            stream_handle,
        )
        return None


@cache_with_registered_key_functions
def make_generate(
    d_out: DeviceArrayLike | IteratorT,
    op: Operator,
):
    """
    Create a generate object that can be called to fill the output with
    values produced by the nullary operation ``op``.

    This is the object-oriented API that allows explicit control over temporary
    storage allocation. For simpler usage, consider using :func:`generate`.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/generate/generate_object.py
           :language: python
           :start-after: # example-begin

    Args:
        d_out: Device array or iterator to store the generated values.
        op: Nullary operation that produces a value.
            The signature is ``() -> T``, where ``T`` is the output data type.

    Returns:
        A callable object that performs the generation.
    """
    op_adapter = make_op_adapter(op)
    return _Generate(d_out, op_adapter)


def generate(
    d_out: DeviceArrayLike | IteratorT,
    op: Operator,
    num_items: int,
    stream=None,
):
    """
    Fills the output with values produced by a nullary generator on the device.

    This function automatically handles temporary storage allocation and execution.

    Equivalent to ``thrust::generate`` in C++: the generator ``op`` is called once
    per output element.  Because the generator runs on the GPU, ``op`` must be a
    ``cuda.compute``-compatible callable (JIT-compiled by numba).

    Example:
        Below, ``generate`` is used to fill an output array with a constant value.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/generate/generate_basic.py
           :language: python
           :start-after: # example-begin

    Args:
        d_out: Device array or iterator to store the generated values.
        op: Nullary operation that produces a value.
            The signature is ``() -> T``, where ``T`` is the output data type.
        num_items: Number of items to generate.
        stream: CUDA stream to use for the operation.
    """
    op_adapter = make_op_adapter(op)
    gen = make_generate(d_out, op_adapter)
    gen(d_out, op_adapter, num_items, stream)
