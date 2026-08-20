# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Multi-threaded contract tests for the STF Python bindings.

These tests assert the documented threading contract (see the "Thread
safety" section of the package README):

* task submission on a shared context is safe from multiple threads;
* wrapper objects (logical_data, task) may be dropped from any thread,
  including concurrently with finalize() on another thread;
* finalize() is idempotent and safe to race with itself (exactly one
  teardown);
* submission on a finalized context raises RuntimeError;
* phase operations (finalize/fence/wait) require quiescing submitters first
  -- the tests quiesce and then assert clean behavior, they do not race
  submission against finalize (that is outside the contract);
* LaunchableGraph.reset() from several threads releases the graph exactly
  once;
* ``with exec_place:`` affinity scopes are per-thread and nestable.

The tests run on any CPython build. On a free-threaded build (3.13t/3.14t)
they exercise true concurrency and also verify that importing the bindings
does not re-enable the GIL.
"""

import os
import sys
import sysconfig
import threading

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402

NUM_THREADS = int(os.environ.get("STF_TEST_NUM_THREADS", "8"))
ITERS = int(os.environ.get("STF_TEST_ITERS", "25"))


def _run_threads(fn, num_threads=NUM_THREADS):
    """Run fn(tid) on num_threads threads after a common barrier; re-raise
    the first worker exception (with all threads joined first)."""
    barrier = threading.Barrier(num_threads)
    errors = []

    def wrapper(tid):
        try:
            barrier.wait()
            fn(tid)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=wrapper, args=(tid,)) for tid in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return errors


def _is_free_threaded_build():
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


# ---------------------------------------------------------------------------
# free-threading declaration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _is_free_threaded_build(),
    reason="only meaningful on a free-threaded CPython build",
)
def test_gil_stays_disabled_after_import():
    """The extension declares freethreading_compatible: importing it must not
    re-enable the GIL.

    Checked in a subprocess so the result is independent of other test
    modules: unrelated third-party extensions imported earlier in the suite
    (e.g. triton via torch) may re-enable the GIL process-wide, which says
    nothing about this package. Any interpreter-level re-enable warning is
    escalated to an error (-W error::RuntimeWarning), so an undeclared
    module in our own import chain fails loudly too.
    """
    import subprocess

    script = (
        "import sys\n"
        "import cuda.stf._experimental._stf_bindings\n"
        "sys.exit(0 if not sys._is_gil_enabled() else 1)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"GIL was re-enabled while importing the bindings:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# concurrent task submission on a shared context
# ---------------------------------------------------------------------------


def test_concurrent_task_submission_shared_ctx():
    """N threads submit read/rw tasks against private and shared logical data
    on one shared context."""
    ctx = stf.context()
    shared = ctx.logical_data(np.zeros(64, dtype=np.float32))

    def worker(tid):
        private = ctx.logical_data(np.ones(64, dtype=np.float32))
        for i in range(ITERS):
            if i % 3 == 0:
                with ctx.task(shared.rw()):
                    pass
            elif i % 3 == 1:
                with ctx.task(shared.read(), private.rw()):
                    pass
            else:
                with ctx.task(private.rw()):
                    pass

    _run_threads(worker)
    ctx.finalize()


def test_concurrent_logical_data_create_destroy():
    """Threads churn logical_data creation/destruction (dropping references
    mid-flight) while submitting tasks on a shared context."""
    ctx = stf.context()

    def worker(tid):
        for i in range(ITERS):
            ld = ctx.logical_data(np.full(32, tid, dtype=np.float32))
            with ctx.task(ld.rw()):
                pass
            if i % 2 == 0:
                del ld  # force wrapper teardown while other threads submit

    _run_threads(worker)
    ctx.finalize()


# ---------------------------------------------------------------------------
# lifetime: wrapper teardown racing finalize, idempotent finalize
# ---------------------------------------------------------------------------


def test_children_dropped_concurrently_with_finalize():
    """Wrapper references may be dropped from any thread at any time --
    including while another thread runs finalize(). The per-context lifetime
    lock must order child destruction against context teardown (no crash,
    no use-after-free)."""
    for _ in range(10):
        ctx = stf.context()
        num_droppers = max(2, NUM_THREADS - 1)
        # Pre-create children (quiescent submission phase).
        buckets = [
            [ctx.logical_data(np.zeros(16, dtype=np.float32)) for _ in range(20)]
            for _ in range(num_droppers)
        ]
        barrier = threading.Barrier(num_droppers + 1)
        errors = []

        def dropper(bucket):
            try:
                barrier.wait()
                while bucket:
                    bucket.pop()  # each pop may run __dealloc__ vs finalize
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=dropper, args=(b,)) for b in buckets
        ]
        for t in threads:
            t.start()
        barrier.wait()
        ctx.finalize()  # races only the droppers, per contract
        for t in threads:
            t.join()
        assert not errors, errors[0]


def test_concurrent_finalize_exactly_once():
    """finalize() racing finalize(): exactly one thread performs the
    teardown, the others no-op; nothing raises."""
    ctx = stf.context()
    ld = ctx.logical_data(np.zeros(16, dtype=np.float32))
    with ctx.task(ld.rw()):
        pass
    del ld

    def worker(tid):
        ctx.finalize()

    _run_threads(worker)
    # And it stays idempotent afterwards.
    ctx.finalize()


def test_submission_after_finalize_raises():
    """Per the phase contract, creating work on a finalized context raises
    RuntimeError instead of touching a destroyed context."""
    ctx = stf.context()
    ld = ctx.logical_data(np.zeros(16, dtype=np.float32))
    with ctx.task(ld.rw()):
        pass
    ctx.finalize()

    with pytest.raises(RuntimeError):
        ctx.task(ld.rw())
    with pytest.raises(RuntimeError):
        ctx.logical_data(np.zeros(16, dtype=np.float32))
    with pytest.raises(RuntimeError):
        ctx.token()


def test_quiesce_then_finalize_from_worker():
    """The supported multi-thread lifecycle: submitters run, are quiesced
    (joined), then one thread -- not necessarily the main one -- finalizes."""
    ctx = stf.context()
    shared = ctx.logical_data(np.zeros(64, dtype=np.float32))

    def worker(tid):
        for _ in range(ITERS):
            with ctx.task(shared.rw()):
                pass

    _run_threads(worker)  # joins all submitters: quiescent now

    t = threading.Thread(target=ctx.finalize)
    t.start()
    t.join()
    with pytest.raises(RuntimeError):
        ctx.task(shared.rw())


# ---------------------------------------------------------------------------
# stackable_context / LaunchableGraph
# ---------------------------------------------------------------------------


def test_launchable_graph_reset_idempotent_and_concurrent_noop():
    """LaunchableGraph.reset() frees the shared reference exactly once.

    The release itself is thread-confined (the stackable scope structure is
    per-thread in the underlying runtime), so the recording thread performs
    it; afterwards, concurrent duplicate reset() calls from other threads
    must all be safe no-ops (the guarded handle swap), and the owning
    context's scope count must have been decremented exactly once so
    finalize() at root remains legal."""
    ctx = stf.stackable_context()
    lx = ctx.logical_data(np.zeros(256, dtype=np.float32))

    ctx.push()
    with ctx.task(lx.rw()):
        pass
    g = ctx.pop_prologue_shared()
    assert g.valid

    g.reset()  # recording thread performs the release
    assert not g.valid
    g.reset()  # idempotent on the same thread

    def worker(tid):
        for _ in range(ITERS):
            g.reset()  # already released: must no-op from any thread

    _run_threads(worker)
    assert not g.valid
    # The scope must have been closed exactly once: finalize() at root works.
    ctx.finalize()


def test_stackable_concurrent_finalize_exactly_once():
    ctx = stf.stackable_context()
    lx = ctx.logical_data(np.zeros(64, dtype=np.float32))
    with ctx.task(lx.rw()):
        pass
    del lx

    def worker(tid):
        ctx.finalize()

    _run_threads(worker)


# ---------------------------------------------------------------------------
# exec_place affinity scopes
# ---------------------------------------------------------------------------


def test_exec_place_scope_nesting_single_thread():
    """Nested ``with place:`` blocks on one thread must restore correctly
    (per-thread scope stack)."""
    place = stf.exec_place.device(0)
    with place:
        with place:
            pass
        # Outer scope must still be open and exit cleanly here.


def test_exec_place_scope_per_thread():
    """Each thread entering the same exec_place gets its own scope."""
    place = stf.exec_place.device(0)

    def worker(tid):
        for _ in range(ITERS):
            with place:
                pass

    _run_threads(worker)


# ---------------------------------------------------------------------------
# PyTorch interop under free-threading
# ---------------------------------------------------------------------------


def test_pytorch_task_multithreaded():
    """pytorch_task from several threads on a shared context. Skipped when
    torch is unavailable (e.g. no free-threaded wheel yet)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA unavailable")
    from cuda.stf._experimental.interop.pytorch import pytorch_task

    ctx = stf.context()
    lds = [
        ctx.logical_data(np.ones(64, dtype=np.float32)) for _ in range(NUM_THREADS)
    ]

    def worker(tid):
        for _ in range(5):
            with pytorch_task(ctx, lds[tid].rw()) as (x,):
                x *= 2.0

    _run_threads(worker)
    out = ctx.wait(lds[0])
    ctx.finalize()
    assert np.allclose(out, 2.0**5)
