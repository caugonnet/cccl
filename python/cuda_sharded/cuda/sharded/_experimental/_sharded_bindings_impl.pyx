# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=False
# cython: freethreading_compatible=True

"""Bindings for the ``cuda::experimental::sharded`` C++ layer.

Design notes
------------
* Containers BIND the C++ implementation -- placement, VMM-backed contiguous
  allocation, and the fixed-size contract live once, in C++.
* Tier-1 algorithms are one Python -> C++ crossing per call; the C++ side owns
  the per-shard loop, the per-place streams, and the cross-place combine.
* Tier-2 (operator-parameterized) starts with standard-op descriptors that
  lower to the pure C++ path. Per-shard interop through
  ``sharded_array.shard(i).__cuda_array_interface__`` is the always-available
  escape hatch for custom operators (numba / cupy / torch per shard).
* Free-threading native: the module declares ``freethreading_compatible`` and
  holds no mutable module-level state. C++ contract errors surface as Python
  exceptions (``std::invalid_argument`` -> ``ValueError``).
"""

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool as cbool
from libcpp.vector cimport vector

import numpy as _np

cdef extern from "sharded_shim.h" namespace "cuda_sharded_shim" nogil:
    cdef cppclass pg_handle:
        pass
    cdef cppclass sa_handle:
        pass
    cdef struct scalar_arg:
        double d
        int64_t i

    const int dtype_f32
    const int dtype_f64
    const int dtype_i32
    const int dtype_i64
    const int reduce_sum
    const int reduce_min
    const int reduce_max
    const int unary_negate
    const int unary_scale
    const int unary_add_scalar
    const int binary_add
    const int binary_mul
    const int binary_axpy

    pg_handle* pg_by_locality_domains(const vector[int]& device_ids) except +
    pg_handle* pg_by_devices(const vector[int]& device_ids) except +
    void pg_destroy(pg_handle* pg)
    size_t pg_size(const pg_handle* pg) except +
    uintptr_t pg_get_stream(pg_handle* pg, size_t place_idx, size_t color) except +
    void pg_sync(pg_handle* pg) except +

    sa_handle* sa_allocate(pg_handle* pg, int dtype, size_t total_size) except +
    sa_handle* sa_allocate_sizes(pg_handle* pg, int dtype, const vector[size_t]& sizes) except +
    sa_handle* sa_allocate_contiguous(pg_handle* pg, int dtype, size_t total_size) except +
    sa_handle* sa_adopt(pg_handle* pg, int dtype, const vector[uintptr_t]& ptrs,
                        const vector[size_t]& sizes, const vector[uintptr_t]& producer_streams) except +
    void sa_destroy(sa_handle* sa)
    int sa_dtype(const sa_handle* sa) except +
    size_t sa_size(const sa_handle* sa) except +
    size_t sa_num_shards(const sa_handle* sa) except +
    cbool sa_is_contiguous(const sa_handle* sa) except +
    uintptr_t sa_contiguous_ptr(const sa_handle* sa) except +
    void sa_shard_info(const sa_handle* sa, size_t idx, uintptr_t* data, size_t* size,
                       size_t* global_offset, uintptr_t* stream) except +
    void sa_copy_from_host(sa_handle* sa, uintptr_t host_ptr) except +
    void sa_copy_to_host(const sa_handle* sa, uintptr_t host_ptr) except +
    void sa_sync(const sa_handle* sa) except +

    void alg_fill(pg_handle* pg, sa_handle* sa, scalar_arg value) except +
    void alg_sequence(pg_handle* pg, sa_handle* sa, scalar_arg start, scalar_arg step) except +
    double alg_reduce_f(pg_handle* pg, const sa_handle* sa, int op) except +
    int64_t alg_reduce_i(pg_handle* pg, const sa_handle* sa, int op) except +
    void alg_inclusive_scan(pg_handle* pg, sa_handle* sa) except +
    void alg_exclusive_scan(pg_handle* pg, sa_handle* sa, scalar_arg init) except +
    void alg_adjacent_difference(pg_handle* pg, sa_handle* input, sa_handle* output) except +
    size_t alg_count(pg_handle* pg, const sa_handle* sa, scalar_arg value) except +
    vector[size_t] alg_histogram_even(pg_handle* pg, const sa_handle* sa, int num_bins,
                                      double lower, double upper) except +
    void alg_sort(pg_handle* pg, sa_handle* sa) except +
    void alg_transform_unary(pg_handle* pg, const sa_handle* input, sa_handle* output,
                             int op, scalar_arg alpha) except +
    void alg_transform_binary(pg_handle* pg, const sa_handle* input1, const sa_handle* input2,
                              sa_handle* output, int op, scalar_arg alpha) except +


# ---------------------------------------------------------------------------
# dtype handling
# ---------------------------------------------------------------------------
# Immutable module-level tables, created once at import: safe under free
# threading (never mutated afterwards).

_DTYPE_TO_TAG = {
    _np.dtype(_np.float32): dtype_f32,
    _np.dtype(_np.float64): dtype_f64,
    _np.dtype(_np.int32): dtype_i32,
    _np.dtype(_np.int64): dtype_i64,
}
_TAG_TO_DTYPE = {tag: dt for dt, tag in _DTYPE_TO_TAG.items()}

_REDUCE_OPS = {"sum": reduce_sum, "min": reduce_min, "max": reduce_max}
_UNARY_OPS = {"negate": unary_negate, "scale": unary_scale, "add_scalar": unary_add_scalar}
_BINARY_OPS = {"add": binary_add, "mul": binary_mul, "axpy": binary_axpy}


cdef int _dtype_tag(object dtype) except -1:
    cdef object dt = _np.dtype(dtype)
    try:
        return _DTYPE_TO_TAG[dt]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {dt}; supported: float32, float64, int32, int64"
        ) from None


cdef scalar_arg _scalar(object value, object dtype):
    """Carry a Python scalar as both a double and an int64; the C++ side picks
    the view matching the array's dtype (64-bit integers do not round-trip
    through double)."""
    cdef scalar_arg s
    if _np.dtype(dtype).kind == "f":
        s.d = float(value)
        s.i = 0
    else:
        s.i = int(value)
        s.d = 0.0
    return s


# ---------------------------------------------------------------------------
# place_group
# ---------------------------------------------------------------------------

cdef class place_group:
    """A group of execution places plus the resources to execute on them
    (per-place stream pools and per-place memory resources).

    Create with :meth:`by_locality_domains` (one place per locality domain of
    each device; devices without locality-domain support contribute a single
    whole-device place) or :meth:`by_devices` (one place per device).

    Thread safety: a ``place_group`` may be shared by multiple threads; stream
    lookup and color assignment are internally synchronized.
    """

    cdef pg_handle* _h

    def __cinit__(self):
        self._h = NULL

    def __dealloc__(self):
        if self._h != NULL:
            pg_destroy(self._h)
            self._h = NULL

    @staticmethod
    cdef place_group _from_handle(pg_handle* h):
        cdef place_group pg = place_group.__new__(place_group)
        pg._h = h
        return pg

    @staticmethod
    def by_locality_domains(device_ids=None):
        """Group with one place per locality domain of every listed device
        (all visible devices when ``device_ids`` is ``None``)."""
        cdef vector[int] ids
        for d in (device_ids or ()):
            ids.push_back(<int> d)
        cdef pg_handle* h
        with nogil:
            h = pg_by_locality_domains(ids)
        return place_group._from_handle(h)

    @staticmethod
    def by_devices(device_ids=None):
        """Group with one place per device: all visible devices, or the listed
        ones."""
        cdef vector[int] ids
        for d in (device_ids or ()):
            ids.push_back(<int> d)
        cdef pg_handle* h
        with nogil:
            h = pg_by_devices(ids)
        return place_group._from_handle(h)

    @property
    def size(self):
        """Number of places in the group."""
        return pg_size(self._h)

    def __len__(self):
        return pg_size(self._h)

    def get_stream(self, size_t place_idx, size_t color=0):
        """CUDA stream handle (as an int) of the ``place_idx``-th place at the
        given stream color."""
        return pg_get_stream(self._h, place_idx, color)

    def sync(self):
        """Synchronize every stream created so far, on every place."""
        with nogil:
            pg_sync(self._h)


# ---------------------------------------------------------------------------
# shard view (the per-shard interop escape hatch)
# ---------------------------------------------------------------------------

cdef class shard_view:
    """A zero-copy view of one shard, consumable by anything that understands
    ``__cuda_array_interface__`` (numba, cupy, torch, ...).

    The view keeps its parent :class:`sharded_array` alive. The interface
    advertises the shard's reference stream, so CAI-aware consumers order
    their work against the same stream the sharded algorithms use.
    """

    cdef readonly object parent          #: the owning sharded_array
    cdef readonly size_t index           #: shard index
    cdef readonly uintptr_t data         #: device pointer
    cdef readonly size_t size            #: number of elements
    cdef readonly size_t global_offset   #: starting index in the logical array
    cdef readonly uintptr_t stream       #: reference CUDA stream handle

    @property
    def dtype(self):
        return self.parent.dtype

    @property
    def __cuda_array_interface__(self):
        d = {
            "shape": (self.size,),
            "typestr": self.parent.dtype.str,
            "data": (self.data, False),
            "strides": None,
            "version": 3,
        }
        if self.stream != 0:
            d["stream"] = self.stream
        return d

    def __repr__(self):
        return (f"shard_view(index={self.index}, size={self.size}, "
                f"global_offset={self.global_offset}, dtype={self.parent.dtype})")


# ---------------------------------------------------------------------------
# sharded_array
# ---------------------------------------------------------------------------

cdef class sharded_array:
    """A 1D array partitioned into placed shards (binding of the C++
    ``cuda::experimental::sharded::sharded_array<T>``).

    Construct with :meth:`allocate`, :meth:`allocate_contiguous`,
    :meth:`adopt`, or :meth:`from_numpy`. Supported dtypes: float32, float64,
    int32, int64.

    Shard sizes are fixed at allocation time. The array keeps a reference to
    the :class:`place_group` it was created from (the group's stream pools
    must outlive the array); adopted arrays additionally keep their source
    buffers alive.

    Thread safety: distinct arrays may be used concurrently from distinct
    threads (including algorithms over them sharing one group); concurrent
    operations on the *same* array are not synchronized.
    """

    cdef sa_handle* _h
    cdef readonly place_group group
    cdef readonly object dtype
    cdef object _keepalive

    def __cinit__(self):
        self._h = NULL

    def __dealloc__(self):
        if self._h != NULL:
            sa_destroy(self._h)
            self._h = NULL

    def __init__(self):
        raise TypeError(
            "use sharded_array.allocate / allocate_contiguous / adopt / from_numpy"
        )

    @staticmethod
    cdef sharded_array _wrap(sa_handle* h, place_group group, object dtype, object keepalive):
        cdef sharded_array sa = sharded_array.__new__(sharded_array)
        sa._h = h
        sa.group = group
        sa.dtype = _np.dtype(dtype)
        sa._keepalive = keepalive
        return sa

    # ---- factories --------------------------------------------------------

    @staticmethod
    def allocate(place_group group, size, dtype=_np.float32):
        """Allocate an owning array over the group's places.

        ``size`` is either a total element count (distributed evenly, the
        remainder going to the first shards) or a sequence of per-shard sizes
        (one per place; shards of size 0 are skipped, which reduces
        ``num_shards``).
        """
        cdef int tag = _dtype_tag(dtype)
        cdef vector[size_t] sizes
        cdef size_t total
        cdef sa_handle* h
        if _np.isscalar(size) or isinstance(size, int):
            total = <size_t> int(size)
            with nogil:
                h = sa_allocate(group._h, tag, total)
        else:
            for s in size:
                sizes.push_back(<size_t> int(s))
            with nogil:
                h = sa_allocate_sizes(group._h, tag, sizes)
        return sharded_array._wrap(h, group, dtype, None)

    @staticmethod
    def allocate_contiguous(place_group group, size_t total_size, dtype=_np.float32):
        """Allocate shards as views into ONE contiguous VA range; each shard's
        bytes are physically owned by its place (VMM backing). The whole array
        is then also readable as one normal device array."""
        cdef int tag = _dtype_tag(dtype)
        cdef sa_handle* h
        with nogil:
            h = sa_allocate_contiguous(group._h, tag, total_size)
        return sharded_array._wrap(h, group, dtype, None)

    @staticmethod
    def adopt(place_group group, buffers):
        """Adopt per-shard device buffers (non-owning view; one buffer per
        group place, in place order).

        Each buffer must expose ``__cuda_array_interface__`` as a 1-D
        contiguous array; all must share one supported dtype. Buffer ``i`` is
        treated as resident at group place ``i``. Producer streams advertised
        by the interface are synchronized before use. The adopted buffers are
        kept alive by the returned array.
        """
        buffers = list(buffers)
        if not buffers:
            raise ValueError("adopt: need at least one buffer")

        cdef vector[uintptr_t] ptrs
        cdef vector[size_t] sizes
        cdef vector[uintptr_t] streams
        dtype = None
        for b in buffers:
            try:
                cai = b.__cuda_array_interface__
            except AttributeError:
                raise TypeError(
                    "adopt: buffers must expose __cuda_array_interface__"
                ) from None
            shape = cai["shape"]
            if len(shape) != 1:
                raise ValueError("adopt: buffers must be 1-D")
            bdt = _np.dtype(cai["typestr"])
            if dtype is None:
                dtype = bdt
            elif bdt != dtype:
                raise ValueError(f"adopt: dtype mismatch ({bdt} vs {dtype})")
            strides = cai.get("strides")
            if strides is not None and tuple(strides) != (dtype.itemsize,):
                raise ValueError("adopt: buffers must be contiguous")
            ptrs.push_back(<uintptr_t> int(cai["data"][0]))
            sizes.push_back(<size_t> int(shape[0]))
            stream = cai.get("stream")
            streams.push_back(<uintptr_t> (0 if stream is None else int(stream)))

        cdef int tag = _dtype_tag(dtype)
        cdef sa_handle* h
        with nogil:
            h = sa_adopt(group._h, tag, ptrs, sizes, streams)
        return sharded_array._wrap(h, group, dtype, buffers)

    @staticmethod
    def from_numpy(place_group group, arr, contiguous=False):
        """Allocate over the group's places (evenly) and copy a 1-D numpy
        array in. Synchronous."""
        arr = _np.ascontiguousarray(arr)
        if arr.ndim != 1:
            raise ValueError("from_numpy: expected a 1-D array")
        if contiguous:
            out = sharded_array.allocate_contiguous(group, arr.shape[0], arr.dtype)
        else:
            out = sharded_array.allocate(group, arr.shape[0], arr.dtype)
        out.copy_from(arr)
        return out

    # ---- host transfer -----------------------------------------------------

    def copy_from(self, arr):
        """Copy a 1-D numpy array of matching size and dtype into the shards.
        Synchronous."""
        arr = _np.ascontiguousarray(arr, dtype=self.dtype)
        if arr.ndim != 1 or arr.shape[0] != self.size:
            raise ValueError(
                f"copy_from: expected a 1-D array of {self.size} elements"
            )
        cdef uintptr_t p = <uintptr_t> arr.ctypes.data
        with nogil:
            sa_copy_from_host(self._h, p)

    def to_numpy(self):
        """Gather the whole logical array into a new numpy array.
        Synchronous."""
        out = _np.empty(self.size, dtype=self.dtype)
        cdef uintptr_t p = <uintptr_t> out.ctypes.data
        with nogil:
            sa_copy_to_host(self._h, p)
        return out

    # ---- introspection -----------------------------------------------------

    @property
    def size(self):
        """Total number of elements across all shards."""
        return sa_size(self._h)

    def __len__(self):
        return sa_size(self._h)

    @property
    def num_shards(self):
        return sa_num_shards(self._h)

    @property
    def is_contiguous(self):
        """True when the whole array is one contiguous VA range."""
        return sa_is_contiguous(self._h)

    @property
    def contiguous_ptr(self):
        """Base device pointer of the contiguous range (0 unless allocated
        with :meth:`allocate_contiguous`)."""
        return sa_contiguous_ptr(self._h)

    def shard(self, size_t index):
        """Zero-copy :class:`shard_view` of one shard -- the per-shard interop
        escape hatch (run numba/cupy/torch kernels on individual shards)."""
        cdef uintptr_t data = 0
        cdef size_t size = 0
        cdef size_t offset = 0
        cdef uintptr_t stream = 0
        sa_shard_info(self._h, index, &data, &size, &offset, &stream)
        cdef shard_view v = shard_view.__new__(shard_view)
        v.parent = self
        v.index = index
        v.data = data
        v.size = size
        v.global_offset = offset
        v.stream = stream
        return v

    def shards(self):
        """List of :class:`shard_view` over all shards."""
        return [self.shard(i) for i in range(self.num_shards)]

    def sync(self):
        """Synchronize every shard's reference stream."""
        with nogil:
            sa_sync(self._h)

    def __repr__(self):
        return (f"sharded_array(size={self.size}, num_shards={self.num_shards}, "
                f"dtype={self.dtype}, contiguous={self.is_contiguous})")


cdef inline sa_handle* _handle(sharded_array a) except NULL:
    if a._h == NULL:
        raise ValueError("operation on an uninitialized sharded_array")
    return a._h


# ---------------------------------------------------------------------------
# Tier-1 algorithms: one crossing per call, C++ owns the per-shard loop,
# the per-place streams, and the cross-place combine. All synchronous.
# ---------------------------------------------------------------------------

def fill(place_group group, sharded_array data, value):
    """Fill every element of ``data`` with ``value``."""
    cdef scalar_arg v = _scalar(value, data.dtype)
    cdef sa_handle* h = _handle(data)
    with nogil:
        alg_fill(group._h, h, v)


def sequence(place_group group, sharded_array data, start=0, step=1):
    """Set ``data[i] = start + i * step`` over the logical index space."""
    cdef scalar_arg a = _scalar(start, data.dtype)
    cdef scalar_arg b = _scalar(step, data.dtype)
    cdef sa_handle* h = _handle(data)
    with nogil:
        alg_sequence(group._h, h, a, b)


def iota(place_group group, sharded_array data, start=0):
    """Set ``data[i] = start + i`` (``sequence`` with step 1)."""
    sequence(group, data, start, 1)


def reduce(place_group group, sharded_array data, op="sum"):
    """Reduce all elements with a standard-op descriptor: ``"sum"``,
    ``"min"`` or ``"max"``. Returns a Python scalar."""
    try:
        opc = _REDUCE_OPS[op]
    except KeyError:
        raise ValueError(
            f"reduce: unknown op {op!r}; expected one of {sorted(_REDUCE_OPS)}"
        ) from None
    cdef int c_op = opc
    cdef sa_handle* h = _handle(data)
    cdef double fres
    cdef int64_t ires
    if data.dtype.kind == "f":
        with nogil:
            fres = alg_reduce_f(group._h, h, c_op)
        return data.dtype.type(fres).item()
    with nogil:
        ires = alg_reduce_i(group._h, h, c_op)
    return int(ires)


def inclusive_scan(place_group group, sharded_array data):
    """In-place inclusive prefix sum across all shards."""
    cdef sa_handle* h = _handle(data)
    with nogil:
        alg_inclusive_scan(group._h, h)


def exclusive_scan(place_group group, sharded_array data, init=0):
    """In-place exclusive prefix sum across all shards, starting at
    ``init``."""
    cdef scalar_arg v = _scalar(init, data.dtype)
    cdef sa_handle* h = _handle(data)
    with nogil:
        alg_exclusive_scan(group._h, h, v)


def adjacent_difference(place_group group, sharded_array input, sharded_array output):
    """Out-of-place adjacent difference: ``output[i] = input[i] - input[i-1]``
    with ``output[0] = input[0]``. Input and output must have matching shard
    layouts and dtypes (``ValueError`` otherwise)."""
    cdef sa_handle* hi = _handle(input)
    cdef sa_handle* ho = _handle(output)
    with nogil:
        alg_adjacent_difference(group._h, hi, ho)


def count(place_group group, sharded_array data, value):
    """Number of elements equal to ``value``."""
    cdef scalar_arg v = _scalar(value, data.dtype)
    cdef sa_handle* h = _handle(data)
    cdef size_t res
    with nogil:
        res = alg_count(group._h, h, v)
    return res


def histogram_even(place_group group, sharded_array data, int num_bins, lower, upper):
    """Histogram with ``num_bins`` equal-width bins over ``[lower, upper)``;
    samples outside the range are ignored. Bounds are cast to the array's
    dtype. Returns a numpy ``uint64`` array of length ``num_bins``."""
    cdef sa_handle* h = _handle(data)
    cdef double lo = float(lower)
    cdef double hi = float(upper)
    cdef vector[size_t] res
    with nogil:
        res = alg_histogram_even(group._h, h, num_bins, lo, hi)
    out = _np.empty(res.size(), dtype=_np.uint64)
    for i in range(res.size()):
        out[i] = res[i]
    return out


def sort(place_group group, sharded_array data):
    """Sort all elements globally, ascending, in place. Shard sizes, offsets
    and capacities are unchanged (contiguous arrays remain valid).

    The engine behind this name is the distributed sort of the multi-GPU
    layer; this binding targets correctness, and its throughput follows the
    engine's (engine improvements land here transparently -- the engine
    slot is a performance detail, not API). Requires one shard per group
    place and all places on a single device (``ValueError`` otherwise).
    """
    cdef sa_handle* h = _handle(data)
    with nogil:
        alg_sort(group._h, h)


# ---------------------------------------------------------------------------
# Tier-2, first rung: transform with standard-op descriptors (pure C++ path).
# Custom Python operators: use the per-shard CAI escape hatch today; a
# JIT-once (numba -> LTO-IR, cached) rung is planned.
# ---------------------------------------------------------------------------

def transform(place_group group, sharded_array input, sharded_array output,
              op, alpha=0):
    """Element-wise unary transform with a standard-op descriptor:

    * ``"negate"``:      ``output[i] = -input[i]``
    * ``"scale"``:       ``output[i] = alpha * input[i]``
    * ``"add_scalar"``:  ``output[i] = input[i] + alpha``

    Input and output must have matching shard layouts and dtypes
    (``ValueError`` otherwise); ``input is output`` is allowed.
    """
    try:
        opc = _UNARY_OPS[op]
    except KeyError:
        raise ValueError(
            f"transform: unknown op {op!r}; expected one of {sorted(_UNARY_OPS)}"
        ) from None
    cdef int c_op = opc
    cdef scalar_arg a = _scalar(alpha, input.dtype)
    cdef sa_handle* hi = _handle(input)
    cdef sa_handle* ho = _handle(output)
    with nogil:
        alg_transform_unary(group._h, hi, ho, c_op, a)


def transform_binary(place_group group, sharded_array input1, sharded_array input2,
                     sharded_array output, op, alpha=0):
    """Element-wise binary transform with a standard-op descriptor:

    * ``"add"``:   ``output[i] = input1[i] + input2[i]``
    * ``"mul"``:   ``output[i] = input1[i] * input2[i]``
    * ``"axpy"``:  ``output[i] = alpha * input1[i] + input2[i]``

    All three arrays must have matching shard layouts and dtypes
    (``ValueError`` otherwise).
    """
    try:
        opc = _BINARY_OPS[op]
    except KeyError:
        raise ValueError(
            f"transform_binary: unknown op {op!r}; expected one of {sorted(_BINARY_OPS)}"
        ) from None
    cdef int c_op = opc
    cdef scalar_arg a = _scalar(alpha, input1.dtype)
    cdef sa_handle* h1 = _handle(input1)
    cdef sa_handle* h2 = _handle(input2)
    cdef sa_handle* ho = _handle(output)
    with nogil:
        alg_transform_binary(group._h, h1, h2, ho, c_op, a)


SUPPORTED_DTYPES = tuple(sorted((str(dt) for dt in _DTYPE_TO_TAG), reverse=True))
