# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

# Skip the whole directory when the compiled bindings are unavailable.
pytest.importorskip("cuda.sharded._experimental._sharded_bindings")

import cuda.sharded._experimental as shd


@pytest.fixture(scope="session")
def group():
    """One place per locality domain of every visible device (falls back to
    one whole-device place per device on machines without domain support)."""
    return shd.place_group.by_locality_domains()


@pytest.fixture(scope="session")
def group3():
    """Three places on device 0 (uneven-split / multi-shard coverage even on
    single-domain machines)."""
    return shd.place_group.by_devices([0, 0, 0])
