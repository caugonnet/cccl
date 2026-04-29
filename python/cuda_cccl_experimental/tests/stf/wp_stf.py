# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Warp-flavoured task context manager for CUDASTF tests/examples.

Same role as ``pytorch_task.py`` but for Warp. Not shipped in the
``cuda.stf`` wheel: this is a thin glue layer that lives next to the
tests so ``cuda.stf`` itself stays free of a Warp dependency.

Why this exists
---------------

Warp + CUDASTF tests in this directory all reinvent the same set of
tiny helpers:

* a ``wp.Stream`` cache keyed on the raw ``cudaStream_t`` (re-registering
  the same raw stream pointer with Warp corrupts its bookkeeping);
* an ``stf_cai`` -> ``wp.array`` adapter that goes through ``ptr=`` (the
  ``data=`` path of ``wp.array`` rejects bare ``__cuda_array_interface__``
  objects);
* a ``ScopedStream(s, sync_enter=False)`` push so that subsequent
  ``wp.empty()`` / ``wp.zeros()`` / ``wp.launch()`` calls without an
  explicit ``stream=`` argument hit the task stream rather than Warp's
  default stream;
* for tasks running inside a *captured* outer scope (a
  ``stackable_context.push()`` block recorded as one
  ``cudaGraph_t``), the additional dance
  ``wp.capture_begin(stream=s, external=True)`` /
  ``wp.capture_end(stream=s)`` so Warp's allocator bookkeeping
  (``g_captures`` / ``g_graph_allocs``) treats the existing capture as
  if Warp had started it -- otherwise ``wp.empty()`` allocates outside
  the graph and sibling tasks alias their scratchpads. This is the
  ``_record_task`` pattern from
  ``newton/examples/mpm/example_mpm_anymal_stf.py``.

Wrapping all four steps once turns the typical Warp+STF body from::

    with ctx.task(lA.read(), lB.rw()) as t:
        s = wrap_stream(t.stream_ptr(), device)
        A = get_arg_warp(t, 0, dtype=wp.float32)
        B = get_arg_warp(t, 1, dtype=wp.float32)
        with wp.ScopedStream(s, sync_enter=False):
            wp.launch(my_kernel, dim=N, inputs=[A, B], stream=s)

into::

    with wp_stf.task(ctx, lA.read(), lB.rw()) as (s, A, B):
        wp.launch(my_kernel, dim=N, inputs=[A, B], stream=s)

and the captured-outer variant from ``example_mpm_anymal_stf.py`` from::

    with ctx.task(tok.write()) as t:
        self._record_task(t, self.simulate_robot)

into::

    with wp_stf.task(ctx, tok.write(), capture=True):
        self.simulate_robot()

A future shipped form of this could live under ``cuda.stf.warp`` (or be
imported as ``warp.stf``); the surface area here is small on purpose so
the move is mechanical.
"""

from __future__ import annotations

import contextlib
import ctypes
from collections.abc import Sequence
from typing import Any

import warp as wp


__all__ = [
    "wrap_stream",
    "as_array",
    "task",
]


# ---------------------------------------------------------------------------
# Capture-detection helpers.
#
# We need three pieces of information to decide whether to wrap a task
# body in ``wp.capture_begin(external=True)`` / ``wp.capture_end``:
#
# 1. Is the task's raw stream currently part of an active CUDA graph
#    capture?  --> ``cudaStreamIsCapturing``.
#
# 2. If it is, what is its ``capture_id``?  Captures forked across
#    streams (the inner stf.context tasks running on top of an outer
#    captured task) all share one ``capture_id``.
#
# 3. Has Warp already registered this ``capture_id`` from an enclosing
#    ``wp.capture_begin``?  Calling ``capture_begin(external=True)``
#    twice for the same ``capture_id`` collides on
#    ``runtime.captures[capture_id]`` and breaks end-cleanup with
#    ``KeyError`` in ``_unregister_capture``. So we only open a new
#    Warp begin/end pair if no enclosing one exists.
#
# (3) is checked via ``wp._src.context.runtime.captures``, which is the
# same dict ``capture_begin`` writes into. That is a private path; if
# Warp ever moves it, this helper updates in one place.
# ---------------------------------------------------------------------------

_CUDART = ctypes.CDLL("libcudart.so")
_CUDART.cudaStreamIsCapturing.argtypes = (
    ctypes.c_void_p,           # cudaStream_t
    ctypes.POINTER(ctypes.c_int),  # cudaStreamCaptureStatus*
)
_CUDART.cudaStreamIsCapturing.restype = ctypes.c_int

# cudaStreamCaptureStatus values
_CSC_NONE = 0
_CSC_ACTIVE = 1


def _stream_is_capturing(raw_ptr: int) -> bool:
    status = ctypes.c_int(0)
    rc = _CUDART.cudaStreamIsCapturing(
        ctypes.c_void_p(int(raw_ptr)), ctypes.byref(status)
    )
    if rc != 0:
        raise RuntimeError(f"cudaStreamIsCapturing failed: rc={rc}")
    return status.value == _CSC_ACTIVE


def _stream_capture_id(raw_ptr: int) -> int:
    """Return the ``capture_id`` for a stream known to be capturing.

    Uses the same backend symbol Warp itself uses
    (``runtime.core.wp_cuda_stream_get_capture_id``).
    """
    import warp._src.context as _wp_ctx  # private but stable
    return int(_wp_ctx.runtime.core.wp_cuda_stream_get_capture_id(int(raw_ptr)))


def _warp_already_tracks_capture(capture_id: int) -> bool:
    """True if Warp has an active ``Graph`` for this ``capture_id`` (i.e.
    an enclosing scope already called ``wp.capture_begin``).
    """
    import warp._src.context as _wp_ctx  # private but stable
    return capture_id in _wp_ctx.runtime.captures


# ---------------------------------------------------------------------------
# wp.Stream cache keyed on raw cudaStream_t.
# ---------------------------------------------------------------------------

_wp_stream_cache: dict[tuple[int, int], wp.Stream] = {}


def wrap_stream(raw_ptr: int, device=None) -> wp.Stream:
    """Return a cached ``wp.Stream`` wrapping ``raw_ptr`` on ``device``.

    Re-registering the same raw ``cudaStream_t`` with Warp corrupts its
    internal stream bookkeeping. STF's stream pool is small, so a
    process-lifetime cache keyed on ``(device, raw_ptr)`` stays small
    too and avoids that footgun.
    """
    if device is None:
        device = wp.get_device()
    key = (id(device), int(raw_ptr))
    s = _wp_stream_cache.get(key)
    if s is None:
        s = wp.Stream(device, cuda_stream=int(raw_ptr))
        _wp_stream_cache[key] = s
    return s


# ---------------------------------------------------------------------------
# stf_cai -> wp.array adapter.
# ---------------------------------------------------------------------------


def _np_to_wp_dtype(np_dtype) -> Any | None:
    import numpy as np

    return wp._src.types.np_dtype_to_warp_type.get(np.dtype(np_dtype))


def as_array(cai, dtype=None, *, shape=None, device=None) -> wp.array:
    """Alias an ``stf_cai`` (returned by ``task.get_arg_cai(i)``) as a
    zero-copy ``wp.array``.

    ``wp.array(data=...)`` rejects raw ``__cuda_array_interface__``
    objects, so we go through ``ptr=`` which maps an external
    allocation without taking ownership. ``dtype`` is inferred from the
    cai if omitted; pass it explicitly for non-trivial mappings (e.g.
    treating a plain byte buffer as a typed view).

    The returned array is only valid while the surrounding
    ``with ctx.task(...)`` (or ``wp_stf.task(...)``) block is active.
    """
    if device is None:
        device = wp.get_device()
    if dtype is None:
        dtype = _np_to_wp_dtype(cai.dtype)
        if dtype is None:
            raise TypeError(
                f"cannot infer Warp dtype from numpy dtype {cai.dtype!r}; "
                f"pass dtype= explicitly"
            )
    cai_shape = tuple(cai.shape) if shape is None else tuple(shape)
    return wp.array(
        ptr=int(cai.ptr),
        dtype=dtype,
        shape=cai_shape,
        device=device,
    )


# ---------------------------------------------------------------------------
# wp_stf.task: combined task + ScopedStream + (optional) external capture
# bookkeeping + zero-copy wp.array views.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def task(
    ctx,
    *deps,
    capture: bool | None = None,
    dtypes: Sequence | None = None,
    device=None,
):
    """Open a CUDASTF task with Warp-friendly conveniences.

    Parameters
    ----------
    ctx
        Any STF context: ``stf.context``, ``stf.stream_ctx``, or a
        ``stf.stackable_context`` (inside or outside a ``push()`` scope).
    *deps
        Forwarded verbatim to ``ctx.task(*deps)``. Each dep may be a
        token access (``tok.read()``/``tok.write()``) or a logical-data
        access (``ld.read()``/``ld.rw()``/...).
    capture
        Whether to wrap the body in
        ``wp.capture_begin(stream=s, external=True)`` /
        ``wp.capture_end`` so Warp's allocator bookkeeping treats the
        outer (already-active) STF capture as one Warp started
        itself -- needed so ``wp.empty()`` / ``wp.zeros()`` inside emit
        ``MEM_ALLOC`` / ``MEM_FREE`` graph nodes rather than allocating
        outside the graph (which would silently alias scratchpads
        between sibling tasks).

        Default ``None`` auto-detects via ``cudaStreamIsCapturing`` on
        the task's stream: ``True`` if the stream is currently part of
        an active capture (i.e. the task is inside a stackable
        ``push()`` scope, directly or transitively through a forked
        inner-context stream), ``False`` otherwise. This is the right
        thing to do in nearly all cases; pass ``capture=False`` only to
        skip the begin/end pair when you know the body issues no
        allocator activity and you want to save the (small) overhead.
    dtypes
        Optional explicit dtype list, parallel to ``deps`` after token
        deps are skipped. If not given, dtypes are inferred from each
        ``stf_cai.dtype``.
    device
        Warp device. Defaults to ``wp.get_device()``.

    Yields
    ------
    tuple ``(stream, *arrays)``
        ``stream`` is a cached ``wp.Stream`` wrapping the task's raw
        ``cudaStream_t`` (and during the ``with`` block it is also
        Warp's active stream, so default-stream allocations and
        launches all hit it). One ``wp.array`` per non-token dep
        follows, in input order.

    Examples
    --------
    Plain task with two array deps -- ``capture=`` auto-resolves to
    ``False`` if the surrounding ``ctx`` is eager, ``True`` if it is
    inside a captured ``push()`` scope::

        with wp_stf.task(ctx, lA.read(), lB.rw()) as (s, A, B):
            wp.launch(my_kernel, dim=N, inputs=[A, B], stream=s)

    Outer task of a captured stackable context (token only)::

        with wp_stf.task(outer_ctx, tok.write()) as (s,):
            self.simulate_step()    # internal wp.empty/wp.launch are captured

    Inner ``stf.context(stream=outer_s)`` task on top of an outer
    captured task -- still auto-detected as captured because the inner
    task's stream forks from the outer capturing stream::

        with wp_stf.task(inner_ctx, tok.write()) as (s,):
            wp.launch(fill_kernel, dim=N, inputs=[v1, val], stream=s)
    """
    if device is None:
        device = wp.get_device()

    t = ctx.task(*deps)
    t.start()

    raw_ptr = int(t.stream_ptr())
    stream = wrap_stream(raw_ptr, device)

    scoped = wp.ScopedStream(stream, sync_enter=False)
    scoped.__enter__()

    # Auto-detect whether to open our own capture_begin/end pair.
    # Skip if either the stream is not in capture, OR an enclosing
    # scope already registered this capture session with Warp (in
    # which case the outer scope's bookkeeping covers our allocs;
    # opening a second ``capture_begin(external=True)`` for the same
    # capture_id collides on ``runtime.captures`` and breaks unwinding).
    if capture is None:
        if _stream_is_capturing(raw_ptr):
            cap_id = _stream_capture_id(raw_ptr)
            capture = not _warp_already_tracks_capture(cap_id)
        else:
            capture = False

    captured = False
    try:
        if capture:
            wp.capture_begin(stream=stream, external=True)
            captured = True

        cais = t.args_cai()
        if cais is None:
            cais_tuple = ()
        elif isinstance(cais, tuple):
            cais_tuple = cais
        else:
            cais_tuple = (cais,)

        if dtypes is not None and len(dtypes) != len(cais_tuple):
            raise ValueError(
                f"dtypes={dtypes!r} has {len(dtypes)} entries but task "
                f"exposes {len(cais_tuple)} non-token dep(s)"
            )

        arrays = [
            as_array(
                cai,
                dtype=None if dtypes is None else dtypes[i],
                device=device,
            )
            for i, cai in enumerate(cais_tuple)
        ]

        yield (stream, *arrays)

    finally:
        try:
            if captured:
                wp.capture_end(stream=stream)
        finally:
            scoped.__exit__(None, None, None)
            t.end()
