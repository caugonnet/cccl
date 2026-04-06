# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

import cuda.stf as stf


def test_ctx():
    ctx = stf.context()
    del ctx


def test_graph_ctx():
    ctx = stf.context(use_graph=True)
    ctx.finalize()


def test_ctx2():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = stf.context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    t = ctx.task(lX.rw())
    t.start()
    t.end()

    t2 = ctx.task(lX.read(), lY.rw())
    t2.start()
    t2.end()

    t3 = ctx.task(lX.read(), lZ.rw())
    t3.start()
    t3.end()

    t4 = ctx.task(lY.read(), lZ.rw())
    t4.start()
    t4.end()

    del ctx


def test_ctx3():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = stf.context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    with ctx.task(lX.rw()):
        pass

    with ctx.task(lX.read(), lY.rw()):
        pass

    with ctx.task(lX.read(), lZ.rw()):
        pass

    with ctx.task(lY.read(), lZ.rw()):
        pass

    del ctx


def test_task_arg_cai_v3():
    X = np.ones(16, dtype=np.float32)

    ctx = stf.context()
    lX = ctx.logical_data(X)

    with ctx.task(lX.read()) as t:
        cai = t.get_arg_cai(0).__cuda_array_interface__
        assert cai["version"] == 3
        assert cai["shape"] == X.shape
        assert cai["typestr"] == X.dtype.str
        assert cai["stream"] == t.stream_ptr()

    ctx.finalize()


def test_logical_data_rejects_non_contiguous():
    arr = np.ones((10, 10), dtype=np.float32)
    strided_view = arr[::2, :]  # non-contiguous: stride along axis 0 != itemsize * shape[1]
    assert not strided_view.flags["C_CONTIGUOUS"]

    ctx = stf.context()
    with pytest.raises(ValueError, match="not contiguous"):
        ctx.logical_data(strided_view)
    ctx.finalize()


if __name__ == "__main__":
    test_ctx3()
