# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
PyTorch-integrated task context manager for STF tests/examples.
Not shipped in the wheel. Use pytorch_task(ctx, *deps) for automatic
PyTorch stream handling and tensor unpacking. Requires PyTorch.
"""

from __future__ import annotations


def pytorch_task(ctx, *args):
    """
    Context manager: ctx.task(*args) with PyTorch stream and tensor conversion.
    Yields tensor(s) from task.tensor_arguments() as a tuple.

    Example
    -------
    >>> from tests.stf.pytorch_task import pytorch_task
    >>> with pytorch_task(ctx, lX.read(), lY.rw()) as (x_tensor, y_tensor):
    ...     y_tensor[:] = x_tensor * 2
    """
    try:
        import torch.cuda as tc
    except ImportError:
        raise RuntimeError(
            "pytorch_task requires PyTorch to be installed. "
            "Install PyTorch or use ctx.task() for a raw task."
        ) from None

    t = ctx.task(*args)

    class _PyTorchTaskContext:
        _stream_ctx = None

        def __enter__(self):
            t.start()
            self._stream_ctx = None
            try:
                stream = tc.ExternalStream(t.stream_ptr())
                self._stream_ctx = tc.stream(stream)
                self._stream_ctx.__enter__()
            except Exception:
                t.end()
                raise
            tensors = t.tensor_arguments()
            if tensors is None:
                return None
            if isinstance(tensors, tuple):
                return tensors
            return (tensors,)

        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                if self._stream_ctx is not None:
                    self._stream_ctx.__exit__(exc_type, exc_val, exc_tb)
            finally:
                t.end()
            return False

    return _PyTorchTaskContext()
