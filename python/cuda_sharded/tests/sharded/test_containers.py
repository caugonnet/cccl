# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Container bindings: place_group, sharded_array allocation/adoption,
round-trips, shard views, and contract errors."""

import numpy as np
import pytest

pytest.importorskip("cuda.sharded._experimental._sharded_bindings")

import cuda.sharded._experimental as shd

DTYPES = [np.float32, np.float64, np.int32, np.int64]


class TestPlaceGroup:
    def test_by_locality_domains(self, group):
        assert group.size >= 1
        assert len(group) == group.size

    def test_by_devices(self):
        g = shd.place_group.by_devices([0])
        assert g.size == 1

    def test_get_stream(self, group):
        s0 = group.get_stream(0)
        assert isinstance(s0, int) and s0 != 0
        # colors index a per-place pool
        assert group.get_stream(0, color=1) != 0

    def test_get_stream_out_of_range(self, group):
        with pytest.raises(ValueError):
            group.get_stream(group.size + 7)

    def test_sync(self, group):
        group.sync()  # no-op is fine; must not raise


class TestAllocate:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_roundtrip(self, group, dtype):
        h = np.arange(1000, dtype=dtype)
        a = shd.sharded_array.from_numpy(group, h)
        assert a.size == 1000
        assert a.dtype == np.dtype(dtype)
        assert a.num_shards == group.size
        assert not a.is_contiguous
        np.testing.assert_array_equal(a.to_numpy(), h)

    def test_even_split_offsets(self, group3):
        a = shd.sharded_array.allocate(group3, 10, np.int32)
        # remainder goes to the first shards: 4, 3, 3
        views = a.shards()
        assert [v.size for v in views] == [4, 3, 3]
        assert [v.global_offset for v in views] == [0, 4, 7]

    def test_explicit_sizes(self, group):
        sizes = [5 + i for i in range(group.size)]
        a = shd.sharded_array.allocate(group, sizes, np.float64)
        assert a.size == sum(sizes)
        assert [v.size for v in a.shards()] == sizes

    def test_sizes_count_mismatch(self, group):
        with pytest.raises(ValueError):
            shd.sharded_array.allocate(group, [4] * (group.size + 1), np.float32)

    def test_unsupported_dtype(self, group):
        with pytest.raises(ValueError):
            shd.sharded_array.allocate(group, 16, np.float16)

    def test_copy_from_wrong_size(self, group):
        a = shd.sharded_array.allocate(group, 32, np.float32)
        with pytest.raises(ValueError):
            a.copy_from(np.zeros(33, dtype=np.float32))


class TestContiguous:
    @pytest.mark.parametrize("dtype", [np.float32, np.int64])
    def test_roundtrip(self, group, dtype):
        h = np.arange(4096, dtype=dtype)
        a = shd.sharded_array.from_numpy(group, h, contiguous=True)
        assert a.is_contiguous
        assert a.contiguous_ptr != 0
        # shard 0 starts exactly at the base pointer; shard views are dense
        views = a.shards()
        assert views[0].data == a.contiguous_ptr
        itemsize = np.dtype(dtype).itemsize
        for v in views:
            assert v.data == a.contiguous_ptr + v.global_offset * itemsize
        np.testing.assert_array_equal(a.to_numpy(), h)

    def test_non_contiguous_has_no_base(self, group):
        a = shd.sharded_array.allocate(group, 64, np.float32)
        assert not a.is_contiguous
        assert a.contiguous_ptr == 0


class TestShardView:
    def test_cai_fields(self, group):
        a = shd.sharded_array.allocate(group, 100, np.float32)
        v = a.shard(0)
        cai = v.__cuda_array_interface__
        assert cai["shape"] == (v.size,)
        assert cai["typestr"] == np.dtype(np.float32).str
        assert cai["data"] == (v.data, False)
        assert cai["strides"] is None
        assert cai["version"] == 3
        assert cai["stream"] == v.stream != 0

    def test_out_of_range(self, group):
        a = shd.sharded_array.allocate(group, 100, np.float32)
        with pytest.raises(IndexError):
            a.shard(a.num_shards)

    def test_keeps_parent_alive(self, group):
        v = shd.sharded_array.allocate(group, 100, np.float32).shard(0)
        assert v.parent.size == 100  # parent survives through the view


class TestAdoptErrors:
    def test_needs_cai(self, group):
        with pytest.raises(TypeError):
            shd.sharded_array.adopt(group, [np.zeros(4, dtype=np.float32)])

    def test_empty(self, group):
        with pytest.raises(ValueError):
            shd.sharded_array.adopt(group, [])
