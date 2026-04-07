# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import cuda.stf as stf


def test_scope_context_manager():
    stf.machine_init()
    place = stf.exec_place.device(0)
    with place:
        pass


def test_scope_nested():
    stf.machine_init()
    outer = stf.exec_place.device(0)
    inner = stf.exec_place.device(0)
    with outer:
        with inner:
            pass


def test_pick_stream():
    stf.machine_init()
    place = stf.exec_place.device(0)
    with place:
        s = place.pick_stream()
        assert isinstance(s, int)
        assert s != 0


def test_affine_data_place():
    place = stf.exec_place.device(0)
    dp = place.affine_data_place
    assert dp.device_id == 0


def test_grid_getitem():
    grid = stf.exec_place_grid.from_devices([0, 0])
    sub = grid[0]
    assert sub.kind == "device"


def test_grid_iteration():
    grid = stf.exec_place_grid.from_devices([0, 0])
    for i in range(grid.size):
        sub = grid[i]
        assert sub.kind == "device"
        assert sub.affine_data_place.device_id == 0


def test_getitem_out_of_bounds():
    place = stf.exec_place.device(0)
    with pytest.raises(IndexError):
        place[1]

    grid = stf.exec_place_grid.from_devices([0, 0])
    with pytest.raises(IndexError):
        grid[grid.size]


def test_machine_init_idempotent():
    stf.machine_init()
    stf.machine_init()


class _PlaceStream:
    """Adapts a raw CUstream pointer (int) to the __cuda_stream__ protocol
    expected by cuda.compute algorithms."""

    def __init__(self, stream_ptr):
        self._ptr = stream_ptr

    def __cuda_stream__(self):
        return (0, self._ptr)


def test_scope_with_cuda_compute():
    """Activate place, pick_stream, run cuda.compute.reduce_into -- no STF tasks."""
    try:
        import cuda.compute
        from cuda.compute import OpKind
    except ImportError:
        pytest.skip("cuda.compute not available")

    import numpy as np

    from cuda.stf._stf_bindings import stf_cai

    stf.machine_init()
    place = stf.exec_place.device(0)

    with place:
        stream_ptr = place.pick_stream()

        n = 1024
        h_input = np.arange(n, dtype=np.float32)

        import numba.cuda

        d_input = numba.cuda.to_device(h_input)
        d_output = numba.cuda.device_array(1, dtype=np.float32)

        input_cai = stf_cai(
            d_input.device_ctypes_pointer.value, (n,), np.float32, stream=stream_ptr
        )
        output_cai = stf_cai(
            d_output.device_ctypes_pointer.value, (1,), np.float32, stream=stream_ptr
        )

        stream = _PlaceStream(stream_ptr)

        h_init = np.array([0.0], dtype=np.float32)
        cuda.compute.reduce_into(
            input_cai,
            output_cai,
            OpKind.PLUS,
            n,
            h_init,
            stream=stream,
        )

        numba.cuda.current_context().synchronize()

        result = d_output.copy_to_host()
        expected = h_input.sum()
        assert abs(result[0] - expected) < 1e-2, f"got {result[0]}, expected {expected}"
