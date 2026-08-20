# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Threaded host dispatch over a shared place_group.

On free-threaded CPython builds (3.13t/3.14t) this doubles as the
free-threading smoke test: the module declares ``freethreading_compatible``,
so importing it must not re-enable the GIL.
"""

import sys
import threading

import numpy as np
import pytest

pytest.importorskip("cuda.sharded._experimental._sharded_bindings")

import cuda.sharded._experimental as shd

N_THREADS = 8
N_ELEMS = 50_000


def test_gil_state():
    """On free-threaded builds, importing the bindings must keep the GIL
    disabled (Py_MOD_GIL_NOT_USED)."""
    if not hasattr(sys, "_is_gil_enabled"):
        pytest.skip("GIL-ful build")
    assert sys._is_gil_enabled() is False


def test_threads_share_group_private_arrays(group):
    """N threads issue tier-1 calls on per-thread arrays over one shared
    group."""
    barrier = threading.Barrier(N_THREADS)
    errors = []

    def worker(tid):
        try:
            barrier.wait()
            a = shd.sharded_array.allocate(group, N_ELEMS, np.int64)
            shd.iota(group, a, start=tid)
            assert shd.reduce(group, a, op="sum") == sum(range(tid, tid + N_ELEMS))
            shd.fill(group, a, tid)
            assert shd.count(group, a, tid) == N_ELEMS
            shd.inclusive_scan(group, a)
            assert shd.reduce(group, a, op="max") == tid * N_ELEMS
            b = shd.sharded_array.allocate(group, N_ELEMS, np.int64)
            shd.transform(group, a, b, "scale", alpha=2)
            assert shd.reduce(group, b, op="max") == 2 * tid * N_ELEMS
        except Exception as e:  # noqa: BLE001
            errors.append((tid, repr(e)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_threads_concurrent_readonly_same_array(group):
    """Read-only tier-1 calls on ONE shared array from many threads."""
    h = np.arange(N_ELEMS, dtype=np.float64)
    a = shd.sharded_array.from_numpy(group, h)
    ref = float(h.sum())
    barrier = threading.Barrier(N_THREADS)
    results = [None] * N_THREADS

    def worker(tid):
        barrier.wait()
        results[tid] = (
            shd.reduce(group, a, op="sum"),
            shd.count(group, a, 123.0),
        )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for total, cnt in results:
        np.testing.assert_allclose(total, ref, rtol=1e-12)
        assert cnt == 1


def test_threads_sort_private_arrays(group):
    """Concurrent sorts on per-thread arrays sharing the group."""
    n_threads = 4
    barrier = threading.Barrier(n_threads)
    errors = []
    rng = np.random.default_rng(1234)
    inputs = [rng.random(10_000).astype(np.float32) for _ in range(n_threads)]

    def worker(tid):
        try:
            barrier.wait()
            a = shd.sharded_array.from_numpy(group, inputs[tid])
            shd.sort(group, a)
            np.testing.assert_array_equal(a.to_numpy(), np.sort(inputs[tid]))
        except Exception as e:  # noqa: BLE001
            errors.append((tid, repr(e)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
