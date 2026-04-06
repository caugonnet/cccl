# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True


from cpython.buffer cimport (
    Py_buffer, PyBUF_FORMAT, PyBUF_ND, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS,
    PyObject_GetBuffer, PyBuffer_Release, PyObject_CheckBuffer
)
from cpython.ref cimport PyObject, Py_INCREF, Py_XDECREF
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.pycapsule cimport (
    PyCapsule_CheckExact, PyCapsule_IsValid, PyCapsule_GetPointer
)
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t, uintptr_t
from libc.string cimport memset, memcpy

import numpy as np

import ctypes
from enum import IntFlag

cdef extern from "<cuda.h>":
    cdef struct OpaqueCUstream_st
    cdef struct OpaqueCUkernel_st
    cdef struct OpaqueCUlibrary_st
    cdef struct OpaqueCUfunc_st

    ctypedef int CUresult
    ctypedef OpaqueCUstream_st *CUstream
    ctypedef OpaqueCUkernel_st *CUkernel
    ctypedef OpaqueCUlibrary_st *CUlibrary
    ctypedef OpaqueCUfunc_st *CUfunction

cdef extern from "<cuda_runtime.h>":
    cdef struct dim3:
        unsigned int x, y, z

cdef extern from "cccl/c/experimental/stf/stf.h":
    #
    # Contexts
    #
    ctypedef struct stf_ctx_handle_t
    ctypedef stf_ctx_handle_t* stf_ctx_handle
    stf_ctx_handle stf_ctx_create()
    stf_ctx_handle stf_ctx_create_graph()
    void stf_ctx_finalize(stf_ctx_handle ctx) nogil
    CUstream stf_fence(stf_ctx_handle ctx) nogil

    #
    # 4D position/dimensions for partition mapping
    #
    ctypedef struct stf_pos4:
        int64_t x
        int64_t y
        int64_t z
        int64_t t

    ctypedef struct stf_dim4:
        uint64_t x
        uint64_t y
        uint64_t z
        uint64_t t

    ctypedef void (*stf_get_executor_fn)(stf_pos4* result, stf_pos4 data_coords, stf_dim4 data_dims, stf_dim4 grid_dims)

    # Forward-declare data place handle (needed by stf_exec_place_set_affine_data_place)
    ctypedef struct stf_data_place_opaque_t
    ctypedef stf_data_place_opaque_t* stf_data_place_handle

    #
    # Exec places (opaque handles)
    #
    ctypedef struct stf_exec_place_opaque_t
    ctypedef stf_exec_place_opaque_t* stf_exec_place_handle
    stf_exec_place_handle stf_exec_place_host()
    stf_exec_place_handle stf_exec_place_device(int dev_id)
    stf_exec_place_handle stf_exec_place_current_device()
    stf_exec_place_handle stf_exec_place_clone(stf_exec_place_handle h)
    void stf_exec_place_destroy(stf_exec_place_handle h)
    int stf_exec_place_is_host(stf_exec_place_handle h)
    int stf_exec_place_is_device(stf_exec_place_handle h)

    # Grid introspection
    void stf_exec_place_get_dims(stf_exec_place_handle h, stf_dim4* out_dims)
    size_t stf_exec_place_size(stf_exec_place_handle h)
    void stf_exec_place_set_affine_data_place(stf_exec_place_handle h, stf_data_place_handle affine_dplace)

    # Grid factories
    stf_exec_place_handle stf_exec_place_grid_from_devices(const int* device_ids, size_t count)
    stf_exec_place_handle stf_exec_place_grid_create(const stf_exec_place_handle* places, size_t count, const stf_dim4* grid_dims)
    void stf_exec_place_grid_destroy(stf_exec_place_handle grid)

    #
    # Data places (functions using the forward-declared handle)
    #
    stf_data_place_handle stf_data_place_host()
    stf_data_place_handle stf_data_place_device(int dev_id)
    stf_data_place_handle stf_data_place_managed()
    stf_data_place_handle stf_data_place_affine()
    stf_data_place_handle stf_data_place_current_device()
    stf_data_place_handle stf_data_place_composite(stf_exec_place_handle grid, stf_get_executor_fn mapper)
    stf_data_place_handle stf_data_place_clone(stf_data_place_handle h)
    void stf_data_place_destroy(stf_data_place_handle h)
    int stf_data_place_get_device_ordinal(stf_data_place_handle h)
    const char* stf_data_place_to_string(stf_data_place_handle h)

    #
    # Logical data
    #
    ctypedef struct stf_logical_data_handle_t
    ctypedef stf_logical_data_handle_t* stf_logical_data_handle
    stf_logical_data_handle stf_logical_data(stf_ctx_handle ctx, void* addr, size_t sz)
    stf_logical_data_handle stf_logical_data_with_place(stf_ctx_handle ctx, void* addr, size_t sz, stf_data_place_handle dplace)
    void stf_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
    void stf_logical_data_destroy(stf_logical_data_handle ld)
    stf_logical_data_handle stf_logical_data_empty(stf_ctx_handle ctx, size_t length)
    stf_logical_data_handle stf_token(stf_ctx_handle ctx)

    #
    # Tasks
    #
    ctypedef struct stf_task_handle_t
    ctypedef stf_task_handle_t* stf_task_handle
    stf_task_handle stf_task_create(stf_ctx_handle ctx)
    void stf_task_set_exec_place(stf_task_handle t, stf_exec_place_handle exec_p)
    void stf_task_set_symbol(stf_task_handle t, const char* symbol)
    void stf_task_add_dep(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
    void stf_task_add_dep_with_dplace(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, stf_data_place_handle data_p)
    void stf_task_start(stf_task_handle t)
    void stf_task_end(stf_task_handle t)
    void stf_task_enable_capture(stf_task_handle t)
    CUstream stf_task_get_custream(stf_task_handle t)
    int stf_task_get_grid_dims(stf_task_handle t, stf_dim4* out_dims)
    int stf_task_get_custream_at_index(stf_task_handle t, size_t place_index, CUstream* out_stream)
    void* stf_task_get(stf_task_handle t, int submitted_index)
    void stf_task_destroy(stf_task_handle t)

    cdef enum stf_access_mode:
        STF_NONE
        STF_READ
        STF_WRITE
        STF_RW

    #
    # CUDA kernel tasks
    #
    ctypedef struct stf_cuda_kernel_handle_t
    ctypedef stf_cuda_kernel_handle_t* stf_cuda_kernel_handle
    stf_cuda_kernel_handle stf_cuda_kernel_create(stf_ctx_handle ctx)
    void stf_cuda_kernel_set_exec_place(stf_cuda_kernel_handle k, stf_exec_place_handle exec_p)
    void stf_cuda_kernel_set_symbol(stf_cuda_kernel_handle k, const char* symbol)
    void stf_cuda_kernel_add_dep(stf_cuda_kernel_handle k, stf_logical_data_handle ld, stf_access_mode m)
    void stf_cuda_kernel_start(stf_cuda_kernel_handle k)
    void* stf_cuda_kernel_get_arg(stf_cuda_kernel_handle k, int index)
    void stf_cuda_kernel_add_desc_cufunc(stf_cuda_kernel_handle k, CUfunction cufunc, dim3 grid_dim_, dim3 block_dim_, size_t shared_mem_, int arg_cnt, const void** args)
    void stf_cuda_kernel_end(stf_cuda_kernel_handle k)
    void stf_cuda_kernel_destroy(stf_cuda_kernel_handle k)

    #
    # Host launch
    #
    ctypedef struct stf_host_launch_handle_t
    ctypedef stf_host_launch_handle_t* stf_host_launch_handle
    ctypedef struct stf_host_launch_deps_handle_t
    ctypedef stf_host_launch_deps_handle_t* stf_host_launch_deps_handle
    ctypedef void (*stf_host_callback_fn)(stf_host_launch_deps_handle deps) noexcept

    stf_host_launch_handle stf_host_launch_create(stf_ctx_handle ctx)
    void stf_host_launch_add_dep(stf_host_launch_handle h, stf_logical_data_handle ld, stf_access_mode m)
    void stf_host_launch_set_symbol(stf_host_launch_handle h, const char* symbol)
    void stf_host_launch_set_user_data(stf_host_launch_handle h, const void* data, size_t size, void (*dtor)(void*))
    void stf_host_launch_submit(stf_host_launch_handle h, stf_host_callback_fn callback)
    void stf_host_launch_destroy(stf_host_launch_handle h)
    void* stf_host_launch_deps_get(stf_host_launch_deps_handle deps, size_t index)
    size_t stf_host_launch_deps_get_size(stf_host_launch_deps_handle deps, size_t index)
    size_t stf_host_launch_deps_size(stf_host_launch_deps_handle deps)
    void* stf_host_launch_deps_get_user_data(stf_host_launch_deps_handle deps)

# ctypes mirror structs for the partition mapper callback.
# The C API uses an out-pointer signature for stf_get_executor_fn:
#   void (*)(stf_pos4* result, stf_pos4 data_coords, stf_dim4 data_dims, stf_dim4 grid_dims)
# This is directly representable as a ctypes CFUNCTYPE.
class _mapper_pos4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int64), ("y", ctypes.c_int64),
                ("z", ctypes.c_int64), ("t", ctypes.c_int64)]

class _mapper_dim4(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint64), ("y", ctypes.c_uint64),
                ("z", ctypes.c_uint64), ("t", ctypes.c_uint64)]

_mapper_cfunc_type = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(_mapper_pos4), _mapper_pos4, _mapper_dim4, _mapper_dim4)


def _make_mapper_callback(mapper):
    """Wrap a Python partitioner as a C function pointer for stf_data_place_composite.

    Returns (callback_object, c_function_pointer_as_int).
    The caller must prevent GC of callback_object for the lifetime of the
    composite data place.
    """
    def _trampoline(result_ptr, c_coords, c_data_dims, c_grid_dims):
        coords = (c_coords.x, c_coords.y, c_coords.z, c_coords.t)
        data_dims = (c_data_dims.x, c_data_dims.y, c_data_dims.z, c_data_dims.t)
        grid_dims = (c_grid_dims.x, c_grid_dims.y, c_grid_dims.z, c_grid_dims.t)
        rx, ry, rz, rt = mapper(coords, data_dims, grid_dims)
        result_ptr[0].x = int(rx)
        result_ptr[0].y = int(ry)
        result_ptr[0].z = int(rz)
        result_ptr[0].t = int(rt)

    callback = _mapper_cfunc_type(_trampoline)
    c_ptr = ctypes.cast(callback, ctypes.c_void_p).value
    return (callback, c_ptr)

class AccessMode(IntFlag):
    NONE  = STF_NONE
    READ  = STF_READ
    WRITE = STF_WRITE
    RW    = STF_RW

class stf_cai:
    """
    Wrapper that exposes CUDA Array Interface v3 for interop (torch, cupy, etc.).
    Supports dict-style access (e.g. obj['data']) for code that expects a CAI dict.
    """
    def __init__(self, ptr, tuple shape, dtype, stream=0):
        self.ptr = ptr               # integer device pointer
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.stream = stream        # CUDA stream handle (int or 0)
        self.__cuda_array_interface__ = {
            'version': 3,
            'shape': self.shape,
            'typestr': self.dtype.str,     # e.g., '<f4' for float32
            'data': (self.ptr, False),     # (ptr, read-only?)
            'strides': None,               # or tuple of strides in bytes
            'stream': self.stream if self.stream != 0 else None,  # CAI v3: 0 disallowed
        }

    def __getitem__(self, key):
        return self.__cuda_array_interface__[key]

    def get(self, key, default=None):
        return self.__cuda_array_interface__.get(key, default)

cdef class logical_data:
    cdef stf_logical_data_handle _ld
    cdef stf_ctx_handle _ctx

    cdef object _dtype
    cdef tuple  _shape
    cdef int    _ndim
    cdef size_t _len
    cdef str    _symbol  # Store symbol for display purposes
    cdef readonly bint _is_token  # readonly makes it accessible from Python

    def __cinit__(self, context ctx=None, object buf=None, data_place dplace=None, shape=None, dtype=None, str name=None):
        cdef Py_buffer view
        cdef int flags

        if ctx is None or buf is None:
            # allow creation via __new__ (eg. in empty_like)
            self._ld = NULL
            self._ctx = NULL
            self._len = 0
            self._dtype = None
            self._shape = ()
            self._ndim = 0
            self._symbol = None
            self._is_token = False
            return

        self._ctx = ctx._ctx
        self._symbol = None  # Initialize symbol
        self._is_token = False  # Initialize token flag

        # Default to host data place if not specified (matches C++ API)
        if dplace is None:
            dplace = data_place.host()

        # Try CUDA Array Interface first
        if hasattr(buf, '__cuda_array_interface__'):
            cai = buf.__cuda_array_interface__

            # Extract CAI information
            data_ptr, readonly = cai['data']
            original_shape = cai['shape']
            typestr = cai['typestr']

            # Handle vector types (e.g., wp.vec2, wp.vec3)
            # Use structured dtype from descr if available
            if typestr.startswith('|V') and 'descr' in cai:
                # Vector/structured type - use descr field
                self._dtype = np.dtype(cai['descr'])
            else:
                # Regular scalar type or vector without descr - use typestr
                self._dtype = np.dtype(typestr)

            # Shape is always the same regardless of type
            self._shape = original_shape

            self._ndim = len(self._shape)

            # Calculate total size in bytes
            itemsize = self._dtype.itemsize
            total_items = 1
            for dim in self._shape:
                total_items *= dim
            self._len = total_items * itemsize

            self._ld = stf_logical_data_with_place(ctx._ctx, <void*><uintptr_t>data_ptr, self._len, dplace._h)
            if self._ld == NULL:
                raise RuntimeError("failed to create logical_data from CUDA array interface")

        else:
            # Fallback to Python buffer protocol; require contiguous memory
            # since STF registers view.buf/view.len as a flat byte range.
            flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_ANY_CONTIGUOUS

            if PyObject_GetBuffer(buf, &view, flags) != 0:
                raise ValueError(
                    "object doesn't support the buffer protocol, is not contiguous, "
                    "or doesn't expose __cuda_array_interface__"
                )

            try:
                self._ndim  = view.ndim
                self._len = view.len
                self._shape = tuple(<Py_ssize_t>view.shape[i] for i in range(view.ndim))
                self._dtype = np.dtype(view.format)
                self._ld = stf_logical_data_with_place(ctx._ctx, view.buf, view.len, dplace._h)
                if self._ld == NULL:
                    raise RuntimeError("failed to create logical_data from buffer")

            finally:
                PyBuffer_Release(&view)

        # Apply symbol name if provided
        if name is not None:
            self.set_symbol(name)


    def set_symbol(self, str name):
        stf_logical_data_set_symbol(self._ld, name.encode())
        self._symbol = name  # Store locally for retrieval

    @property
    def symbol(self):
        """Get the symbol name of this logical data, if set."""
        return self._symbol

    def __dealloc__(self):
        if self._ld != NULL:
            try:
                stf_logical_data_destroy(self._ld)
            except Exception as e:
                print(f"stf.logical_data: cleanup failed: {e}")
            self._ld = NULL

    def __repr__(self):
        """Return a detailed string representation of the logical_data object."""
        return (f"logical_data(shape={self._shape}, dtype={self._dtype}, "
                f"is_token={self._is_token}, symbol={self._symbol!r}, "
                f"len={self._len}, ndim={self._ndim})")

    @property
    def dtype(self):
        """Return the dtype of the logical data."""
        return self._dtype

    @property
    def shape(self):
        """Return the shape of the logical data."""
        return self._shape

    def read(self, dplace=None):
        return dep(self, AccessMode.READ.value, dplace)

    def write(self, dplace=None):
        return dep(self, AccessMode.WRITE.value, dplace)

    def rw(self, dplace=None):
        return dep(self, AccessMode.RW.value, dplace)

    def empty_like(self):
        """
        Create a new logical_data with the same shape (and dtype metadata)
        as this object.
        """
        if self._ld == NULL:
            raise RuntimeError("source logical_data handle is NULL")

        cdef logical_data out = logical_data.__new__(logical_data)
        out._ld = stf_logical_data_empty(self._ctx, self._len)
        if out._ld == NULL:
            raise RuntimeError("failed to create empty logical_data")
        out._ctx   = self._ctx
        out._dtype = self._dtype
        out._shape = self._shape
        out._ndim  = self._ndim
        out._len   = self._len
        out._symbol = None
        out._is_token = False

        return out

    @staticmethod
    def token(context ctx):
        cdef logical_data out = logical_data.__new__(logical_data)
        out._ctx   = ctx._ctx
        out._dtype = None
        out._shape = None
        out._ndim  = 0
        out._len   = 0
        out._symbol = None  # New object has no symbol initially
        out._is_token = True
        out._ld = stf_token(ctx._ctx)
        if out._ld == NULL:
            raise RuntimeError("failed to create STF token")

        return out

    @staticmethod
    def init_by_shape(context ctx, shape, dtype, str name=None):
        """
        Create a new logical_data from a shape and a dtype.
        """
        try:
            shape_tuple = tuple(int(dim) for dim in shape)
        except TypeError:
            raise TypeError("shape must be an iterable of integers")
        if not shape_tuple:
            raise ValueError("shape must contain at least one dimension")
        for dim in shape_tuple:
            if dim <= 0:
                raise ValueError("all shape dimensions must be positive integers")
        cdef logical_data out = logical_data.__new__(logical_data)
        out._ctx   = ctx._ctx
        out._dtype = np.dtype(dtype)
        out._shape = shape_tuple
        out._ndim  = len(shape_tuple)
        cdef size_t total_items = 1
        for dim in shape_tuple:
            total_items *= dim
        out._len   = total_items * out._dtype.itemsize
        out._symbol = None
        out._is_token = False
        out._ld = stf_logical_data_empty(ctx._ctx, out._len)
        if out._ld == NULL:
            raise RuntimeError("failed to create logical_data from shape")

        if name is not None:
            out.set_symbol(name)

        return out

    def borrow_ctx_handle(self):
        ctx = context(borrowed=True)
        ctx.borrow_from_handle(self._ctx)
        return ctx

class dep:
    __slots__ = ("ld", "mode", "dplace")
    def __init__(self, logical_data ld, int mode, dplace=None):
        self.ld   = ld
        self.mode = mode
        self.dplace = dplace  # can be None or a data place
    def __iter__(self):      # nice unpacking support
        yield self.ld
        yield self.mode
        yield self.dplace
    def __repr__(self):
        return f"dep({self.ld!r}, {self.mode}, {self.dplace!r})"
    def get_ld(self):
        return self.ld

def read(ld, dplace=None):   return dep(ld, AccessMode.READ.value, dplace)
def write(ld, dplace=None):  return dep(ld, AccessMode.WRITE.value, dplace)
def rw(ld, dplace=None):     return dep(ld, AccessMode.RW.value, dplace)

cdef class exec_place:
    cdef stf_exec_place_handle _h

    def __cinit__(self):
        self._h = NULL

    def __dealloc__(self):
        if self._h != NULL:
            try:
                stf_exec_place_destroy(self._h)
            except Exception as e:
                print(f"stf.exec_place: cleanup failed: {e}")
            self._h = NULL

    @staticmethod
    def device(int dev_id):
        cdef exec_place p = exec_place.__new__(exec_place)
        p._h = stf_exec_place_device(dev_id)
        if p._h == NULL:
            raise RuntimeError(f"failed to create exec_place for device {dev_id}")
        return p

    @staticmethod
    def host():
        cdef exec_place p = exec_place.__new__(exec_place)
        p._h = stf_exec_place_host()
        if p._h == NULL:
            raise RuntimeError("failed to create host exec_place")
        return p

    @staticmethod
    def current_device():
        cdef exec_place p = exec_place.__new__(exec_place)
        p._h = stf_exec_place_current_device()
        if p._h == NULL:
            raise RuntimeError("failed to create current_device exec_place")
        return p

    @property
    def kind(self) -> str:
        if stf_exec_place_is_host(self._h):
            return "host"
        return "device"

    @property
    def dims(self):
        """Grid dimensions as (x, y, z, t). Scalar places return (1, 1, 1, 1)."""
        cdef stf_dim4 d
        stf_exec_place_get_dims(self._h, &d)
        return (d.x, d.y, d.z, d.t)

    @property
    def size(self):
        """Number of sub-places (1 for scalar places)."""
        return stf_exec_place_size(self._h)

    def set_affine_data_place(self, data_place dplace):
        """Set the affine data place for this exec place grid.

        Dependencies using ``data_place.affine()`` will resolve to ``dplace``
        when this exec place is used as the task's execution place.
        """
        stf_exec_place_set_affine_data_place(self._h, dplace._h)


cdef class exec_place_grid(exec_place):
    """Grid of execution places (a subclass of exec_place).

    Use wherever an exec_place is expected.  Create with ``from_devices()``
    or ``create()``.
    """
    cdef object _mapper_keep_alive  # prevent GC of ctypes callback if mapper was set

    def __cinit__(self):
        self._mapper_keep_alive = None

    @staticmethod
    def from_devices(device_ids):
        """Create a 1-D grid with one place per device.

        Parameters
        ----------
        device_ids : sequence of int
            Device ordinals (e.g. ``[0, 1]`` for two GPUs, or ``[0, 0]``
            for the same device repeated).
        """
        cdef int c_ids[64]
        cdef size_t n = len(device_ids)
        if n == 0:
            raise ValueError("device_ids must contain at least one device")
        if n > 64:
            raise ValueError("at most 64 devices supported")
        for i in range(n):
            c_ids[i] = int(device_ids[i])
        cdef exec_place_grid g = exec_place_grid.__new__(exec_place_grid)
        g._h = stf_exec_place_grid_from_devices(c_ids, n)
        if g._h == NULL:
            raise RuntimeError("failed to create exec_place grid from devices")
        return g

    @staticmethod
    def create(places, grid_dims=None, mapper=None):
        """Create a grid from a list of exec_place objects.

        Parameters
        ----------
        places : list of exec_place
            Individual execution places that form the grid.
        grid_dims : tuple of int, optional
            Shape of the grid as ``(x, y, z, t)``.  If *None*, a 1-D
            grid of length ``len(places)`` is used.
        mapper : callable, optional
            If provided, a composite data place is created from this
            partitioner and set as the grid's affine data place so that
            dependencies with ``data_place.affine()`` resolve automatically.
            Signature: ``(data_coords, data_dims, grid_dims) -> (x, y, z, t)``.
        """
        cdef size_t n = len(places)
        if n == 0:
            raise ValueError("places must contain at least one place")
        if n > 64:
            raise ValueError("at most 64 places supported")

        cdef stf_exec_place_handle c_places[64]
        cdef stf_dim4 dims
        cdef exec_place ep

        converted = []
        for i in range(n):
            ep = <exec_place?>places[i]
            converted.append(ep)
            c_places[i] = ep._h

        cdef exec_place_grid g = exec_place_grid.__new__(exec_place_grid)
        if grid_dims is not None:
            dims.x = int(grid_dims[0])
            dims.y = int(grid_dims[1]) if len(grid_dims) > 1 else 1
            dims.z = int(grid_dims[2]) if len(grid_dims) > 2 else 1
            dims.t = int(grid_dims[3]) if len(grid_dims) > 3 else 1
            g._h = stf_exec_place_grid_create(c_places, n, &dims)
        else:
            g._h = stf_exec_place_grid_create(c_places, n, NULL)

        if g._h == NULL:
            raise RuntimeError("failed to create exec_place grid")

        if mapper is not None:
            dplace = data_place.composite(g, mapper)
            g.set_affine_data_place(dplace)
            g._mapper_keep_alive = dplace

        return g


cdef class data_place:
    cdef stf_data_place_handle _h
    cdef object _mapper_callback  # prevent GC of ctypes callback for composite places

    def __cinit__(self):
        self._h = NULL
        self._mapper_callback = None

    def __dealloc__(self):
        if self._h != NULL:
            try:
                stf_data_place_destroy(self._h)
            except Exception as e:
                print(f"stf.data_place: cleanup failed: {e}")
            self._h = NULL

    @staticmethod
    def device(int dev_id):
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_device(dev_id)
        if p._h == NULL:
            raise RuntimeError(f"failed to create data_place for device {dev_id}")
        return p

    @staticmethod
    def host():
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_host()
        if p._h == NULL:
            raise RuntimeError("failed to create host data_place")
        return p

    @staticmethod
    def managed():
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_managed()
        if p._h == NULL:
            raise RuntimeError("failed to create managed data_place")
        return p

    @staticmethod
    def affine():
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_affine()
        if p._h == NULL:
            raise RuntimeError("failed to create affine data_place")
        return p

    @staticmethod
    def current_device():
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_current_device()
        if p._h == NULL:
            raise RuntimeError("failed to create current_device data_place")
        return p

    @staticmethod
    def composite(exec_place grid, object mapper):
        """Create a composite data place: grid of execution places + partition function.

        The partitioner (mapper) is a callable with signature::

            (data_coords, data_dims, grid_dims) -> (x, y, z, t)

        Each argument/return is a 4-tuple of integers:

        - *data_coords*: logical position in the data
        - *data_dims*: full shape of the data
        - *grid_dims*: shape of the execution place grid
        - return: position in the grid (which place owns this data element)

        Example — blocked partition along first dimension::

            def blocked_1d(data_coords, data_dims, grid_dims):
                n = data_dims[0]
                nplaces = grid_dims[0]
                part_size = max((n + nplaces - 1) // nplaces, 1)
                place_x = min(data_coords[0] // part_size, nplaces - 1)
                return (place_x, 0, 0, 0)

            grid = exec_place_grid.from_devices([0, 1])
            dplace = data_place.composite(grid, blocked_1d)
        """
        if not callable(mapper):
            raise TypeError(
                "mapper must be callable: (data_coords, data_dims, grid_dims) -> (x, y, z, t)")
        callback_obj, c_ptr = _make_mapper_callback(mapper)
        cdef data_place p = data_place.__new__(data_place)
        p._mapper_callback = callback_obj
        cdef uintptr_t ptr_val = c_ptr
        p._h = stf_data_place_composite(grid._h, <stf_get_executor_fn>ptr_val)
        if p._h == NULL:
            raise RuntimeError("failed to create composite data_place")
        return p

    @property
    def kind(self) -> str:
        cdef const char* s = stf_data_place_to_string(self._h)
        return s.decode("utf-8") if s != NULL else "unknown"

    @property
    def device_id(self) -> int:
        return stf_data_place_get_device_ordinal(self._h)



cdef class task:
    cdef stf_task_handle _t

    # list of logical data in deps: we need this because we can't exchange
    # dtype/shape easily through the C API of STF
    cdef list _lds_args

    def __cinit__(self, context ctx):
        self._t = stf_task_create(ctx._ctx)
        if self._t == NULL:
            raise RuntimeError("failed to create STF task")
        self._lds_args = []

    def __dealloc__(self):
        if self._t != NULL:
            try:
                stf_task_destroy(self._t)
            except Exception as e:
                print(f"stf.task: cleanup failed: {e}")

    def start(self):
        # This is ignored if this is not a graph task
        stf_task_enable_capture(self._t)

        stf_task_start(self._t)

    def end(self):
        stf_task_end(self._t)

    def add_dep(self, object d):
        """
        Accept a `dep` instance created with read(ld), write(ld), or rw(ld).
        """
        if not isinstance(d, dep):
            raise TypeError("add_dep expects read(ld), write(ld) or rw(ld)")

        cdef logical_data ldata = <logical_data> d.ld
        cdef int           mode_int  = int(d.mode)
        cdef stf_access_mode mode_ce = <stf_access_mode> mode_int
        cdef data_place dp

        if d.dplace is None:
            stf_task_add_dep(self._t, ldata._ld, mode_ce)
        else:
            dp = <data_place> d.dplace
            stf_task_add_dep_with_dplace(self._t, ldata._ld, mode_ce, dp._h)

        self._lds_args.append(ldata)

    def set_symbol(self, str name):
        stf_task_set_symbol(self._t, name.encode())

    def set_exec_place(self, object exec_p):
       if not isinstance(exec_p, exec_place):
           raise TypeError("set_exec_place expects and exec_place argument")

       cdef exec_place ep = <exec_place> exec_p
       stf_task_set_exec_place(self._t, ep._h)

    def stream_ptr(self) -> int:
        """
        Return the raw CUstream pointer as a Python int
        (memory address).  Suitable for ctypes or PyCUDA.
        """
        cdef CUstream s = stf_task_get_custream(self._t)
        return <uintptr_t> s         # cast pointer -> Py int

    def get_grid_dims(self):
        """When the task's exec place is a grid, return (x, y, z, t) shape.

        Call after start(). Returns None if the task is not on a grid.
        """
        cdef stf_dim4 dims
        if stf_task_get_grid_dims(self._t, &dims) != 0:
            return None
        return (dims.x, dims.y, dims.z, dims.t)

    def get_stream_at_index(self, size_t place_index):
        """When the task's exec place is a grid, return the CUstream for the
        given linear index (0 to product of grid dims - 1) as a Python int.

        Call after start(). Raises if not a grid or index invalid.
        """
        cdef CUstream s
        if stf_task_get_custream_at_index(self._t, place_index, &s) != 0:
            raise RuntimeError("task is not on a grid or place_index out of range")
        return <uintptr_t> s

    def get_stream_ptrs(self):
        """Return a list of raw CUstream pointers (as ints), one per place in the grid.

        Convenience for grid tasks. Returns [stream_ptr()] (length 1) for non-grid tasks.
        Call after start().
        """
        dims = self.get_grid_dims()
        if dims is None:
            return [self.stream_ptr()]
        cdef size_t n = dims[0] * dims[1] * dims[2] * dims[3]
        return [self.get_stream_at_index(i) for i in range(n)]

    def get_arg(self, index) -> int:
        if self._lds_args[index]._is_token:
           raise RuntimeError("cannot materialize a token argument")

        cdef void *ptr = stf_task_get(self._t, index)
        return <uintptr_t>ptr

    def get_arg_cai(self, index):
        """Return the argument as a CUDA Array Interface v3 object.
        The returned view is only valid while the task is active, i.e. until stf_task_end()
        or the end of the surrounding ``with ctx.task(...)`` block."""
        ptr = self.get_arg(index)
        return stf_cai(ptr, self._lds_args[index].shape, self._lds_args[index].dtype, stream=self.stream_ptr())

    def args_cai(self):
        """
        Return all non-token buffer arguments as CUDA Array Interface v3 objects.
        Returns None, a single object, or a tuple. Use from non-shipped code (e.g. tests) to
        convert to numba/torch/cupy via from_cuda_array_interface or torch.as_tensor(obj).
        Returned views are only valid while the task is active.
        """
        non_token_cais = [self.get_arg_cai(i) for i in range(len(self._lds_args))
                          if not self._lds_args[i]._is_token]

        if len(non_token_cais) == 0:
            return None
        elif len(non_token_cais) == 1:
            return non_token_cais[0]
        return tuple(non_token_cais)

    # ---- context‑manager helpers -------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, object exc_type, object exc, object tb):
        """
        Always called, even if an exception occurred inside the block.
        """
        self.end()
        return False

cdef dim3 _to_dim3(object val):
    """Convert an int or 1-3 element tuple to a dim3 struct."""
    cdef dim3 d
    cdef tuple t
    cdef int n
    if isinstance(val, int):
        d.x = val; d.y = 1; d.z = 1
        return d
    t = tuple(val)
    n = len(t)
    if n == 1:
        d.x = t[0]; d.y = 1; d.z = 1
    elif n == 2:
        d.x = t[0]; d.y = t[1]; d.z = 1
    elif n == 3:
        d.x = t[0]; d.y = t[1]; d.z = t[2]
    else:
        raise ValueError("grid/block must have 1-3 dimensions")
    return d


cdef class cuda_kernel:
    """Optimized CUDA kernel task with full dependency tracking.

    Unlike a generic ``task`` where the user manually launches work on a
    stream, ``cuda_kernel`` receives the complete kernel description
    (function, grid, block, args) so STF can create native CUDA graph
    kernel nodes, avoiding stream-capture overhead.
    """
    cdef stf_cuda_kernel_handle _k
    cdef list _lds_args
    cdef list _arg_holders  # keep ParamHolder(s) alive until end()

    def __cinit__(self, context ctx):
        self._k = stf_cuda_kernel_create(ctx._ctx)
        if self._k == NULL:
            raise RuntimeError("failed to create STF cuda_kernel")
        self._lds_args = []
        self._arg_holders = []

    def __dealloc__(self):
        if self._k != NULL:
            try:
                stf_cuda_kernel_destroy(self._k)
            except Exception as e:
                print(f"stf.cuda_kernel: cleanup failed: {e}")

    def start(self):
        stf_cuda_kernel_start(self._k)

    def end(self):
        stf_cuda_kernel_end(self._k)
        self._arg_holders.clear()

    def add_dep(self, object d):
        if not isinstance(d, dep):
            raise TypeError("add_dep expects read(ld), write(ld) or rw(ld)")
        cdef logical_data ldata = <logical_data>d.ld
        cdef int mode_int = int(d.mode)
        cdef stf_access_mode mode_ce = <stf_access_mode>mode_int
        stf_cuda_kernel_add_dep(self._k, ldata._ld, mode_ce)
        self._lds_args.append(ldata)

    def set_symbol(self, str name):
        stf_cuda_kernel_set_symbol(self._k, name.encode())

    def set_exec_place(self, object exec_p):
        if not isinstance(exec_p, exec_place):
            raise TypeError("set_exec_place expects an exec_place argument")
        cdef exec_place ep = <exec_place>exec_p
        stf_cuda_kernel_set_exec_place(self._k, ep._h)

    def get_arg(self, int index) -> int:
        if self._lds_args[index]._is_token:
            raise RuntimeError("cannot materialize a token argument")
        cdef void* ptr = stf_cuda_kernel_get_arg(self._k, index)
        return <uintptr_t>ptr

    def get_arg_cai(self, int index):
        ptr = self.get_arg(index)
        return stf_cai(ptr, self._lds_args[index].shape, self._lds_args[index].dtype)

    def launch(self, kernel, grid, block, args, size_t shmem=0):
        """Launch a CUDA kernel through STF.

        Parameters
        ----------
        kernel : cuda.core.Kernel or int
            Compiled kernel object (``cuda.core.Kernel``) or raw
            ``CUfunction`` handle as an integer.
        grid : int or tuple
            Grid dimensions (up to 3D).
        block : int or tuple
            Block dimensions (up to 3D).
        args : list
            Kernel arguments.  ``int`` values are treated as device
            pointers (matching ``cuda.core.launch`` conventions);
            use ``ctypes`` or ``numpy`` scalars for typed values.
        shmem : int, optional
            Dynamic shared memory in bytes (default 0).
        """
        from cuda.core._kernel_arg_handler import ParamHolder

        cdef uintptr_t func_handle
        if hasattr(kernel, '_handle'):
            handle = kernel._handle
            try:
                from cuda.bindings.driver import CUkernel as _CUkernel
                if isinstance(handle, _CUkernel):
                    from cuda.bindings.driver import cuKernelGetFunction
                    err, cufunc = cuKernelGetFunction(handle)
                    if int(err) != 0:
                        raise RuntimeError(
                            f"cuKernelGetFunction failed with error {err}")
                    func_handle = <uintptr_t>int(cufunc)
                else:
                    func_handle = <uintptr_t>int(handle)
            except ImportError:
                func_handle = <uintptr_t>int(handle)
        else:
            func_handle = <uintptr_t>int(kernel)

        cdef dim3 grid_dim = _to_dim3(grid)
        cdef dim3 block_dim = _to_dim3(block)

        holder = ParamHolder(tuple(args))
        cdef const void** raw_args = <const void**><uintptr_t>(holder.ptr)

        stf_cuda_kernel_add_desc_cufunc(
            self._k, <CUfunction>func_handle,
            grid_dim, block_dim, shmem,
            <int>len(args), raw_args)

        self._arg_holders.append(holder)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, object exc_type, object exc, object tb):
        self.end()
        return False


# ---------------------------------------------------------------------------
# host_launch helpers: C callback trampoline and Python payload destructor
# ---------------------------------------------------------------------------

cdef void _python_payload_destructor(void* data) noexcept with gil:
    """Release the Python payload tuple when C++ destroys the host_launch scope."""
    cdef PyObject* obj = (<PyObject**>data)[0]
    Py_XDECREF(obj)

cdef void _host_launch_trampoline(stf_host_launch_deps_handle deps_h) noexcept with gil:
    """C callback that unpacks deps as numpy arrays and calls the Python fn."""
    cdef PyObject** payload_ptr_ptr = <PyObject**>stf_host_launch_deps_get_user_data(deps_h)
    cdef object payload = <object>(payload_ptr_ptr[0])
    fn, user_args, dep_meta = payload

    cdef size_t ndeps = stf_host_launch_deps_size(deps_h)
    dep_arrays = []
    cdef size_t i
    cdef void* ptr
    cdef size_t nbytes
    for i in range(ndeps):
        ptr = stf_host_launch_deps_get(deps_h, i)
        nbytes = stf_host_launch_deps_get_size(deps_h, i)
        shape, dtype = dep_meta[i]
        dt = np.dtype(dtype)
        cbuf = (ctypes.c_char * nbytes).from_address(<uintptr_t>ptr)
        arr = np.frombuffer(cbuf, dtype=dt).reshape(shape)
        dep_arrays.append(arr)

    fn(*dep_arrays, *user_args)

cdef class context:
    cdef stf_ctx_handle _ctx
    # Is this a context that we have borrowed ?
    cdef bint _borrowed

    def __cinit__(self, bint use_graph=False, bint borrowed=False):
        self._ctx = <stf_ctx_handle>NULL
        self._borrowed = borrowed
        if not borrowed:
            if use_graph:
                self._ctx = stf_ctx_create_graph()
            else:
                self._ctx = stf_ctx_create()
            if self._ctx == NULL:
                raise RuntimeError("failed to create STF context")

    cdef borrow_from_handle(self, stf_ctx_handle ctx_handle):
        if self._ctx != NULL:
            raise RuntimeError("context already initialized")

        if not self._borrowed:
            raise RuntimeError("cannot call borrow_from_handle on this context")

        self._ctx = ctx_handle

    def __repr__(self):
        return f"context(handle={<uintptr_t>self._ctx}, borrowed={self._borrowed})"

    def __dealloc__(self):
        if not self._borrowed:
            try:
                self.finalize()
            except Exception as e:
                print(f"stf.context: cleanup failed: {e}")

    def finalize(self):
        if self._borrowed:
            raise RuntimeError("cannot finalize borrowed context")

        cdef stf_ctx_handle h = self._ctx
        if h != NULL:
            self._ctx = NULL
            with nogil:
                stf_ctx_finalize(h)
        else:
            self._ctx = NULL

    def fence(self):
        """Return a CUDA stream that completes when all pending tasks finish.

        Provides a non-blocking synchronization point: the returned stream
        will be signaled once every task submitted so far has completed.
        Unlike ``finalize()``, this does **not** destroy the context, so
        more tasks can be submitted afterwards.

        Returns
        -------
        int
            Raw ``CUstream`` handle as a Python integer (suitable for
            ``cudaStreamSynchronize`` via ctypes, PyCUDA, etc.).

        Examples
        --------
        >>> ctx = stf.context()
        >>> ld = ctx.logical_data(np.zeros(8, dtype=np.float32))
        >>> with ctx.task(ld.rw()):
        ...     pass
        >>> stream = ctx.fence()
        >>> # cudaStreamSynchronize(stream) to wait for completion
        >>> ctx.finalize()
        """
        if self._ctx == NULL:
            raise RuntimeError("context handle is NULL")
        cdef CUstream s
        with nogil:
            s = stf_fence(self._ctx)
        return <uintptr_t>s

    def logical_data(self, object buf, data_place dplace=None, str name=None):
        """
        Create and return a `logical_data` object bound to this context [PRIMARY API].

        This is the primary function for creating logical data from existing buffers.
        It supports both Python buffer protocol objects and CUDA Array Interface objects,
        with explicit data_place specification for optimal STF data movement strategies.

        Parameters
        ----------
        buf : any buffer‑supporting Python object or __cuda_array_interface__ object
              (NumPy array, Warp array, CuPy array, bytes, bytearray, memoryview, …)
        dplace : data_place, optional
              Specifies where the buffer is located (host, device, managed, affine).
              Defaults to data_place.host() for backward compatibility.
              Essential for GPU arrays - use data_place.device() for optimal performance.
        name : str, optional
              Symbol name for debugging and DOT graph output.

        Examples
        --------
        >>> # Host memory (explicit - recommended)
        >>> host_place = data_place.host()
        >>> ld = ctx.logical_data(numpy_array, host_place)
        >>>
        >>> # GPU device memory (recommended for CUDA arrays)
        >>> device_place = data_place.device(0)
        >>> ld = ctx.logical_data(warp_array, device_place)
        >>>
        >>> # With a symbol name for debugging
        >>> ld = ctx.logical_data(numpy_array, name="X")
        >>>
        >>> # Backward compatibility (defaults to host)
        >>> ld = ctx.logical_data(numpy_array)  # Same as specifying host

        Note
        ----
        For GPU arrays (Warp, CuPy, etc.), always specify data_place.device()
        for zero-copy performance and correct memory management.
        """
        return logical_data(self, buf, dplace, name=name)


    def logical_data_empty(self, shape, dtype=None, str name=None):
        """
        Create logical data with uninitialized values.

        Equivalent to numpy.empty() but for STF logical data.

        Parameters
        ----------
        shape : tuple
            Shape of the array
        dtype : numpy.dtype, optional
            Data type. Defaults to np.float64.
        name : str, optional
            Symbol name for debugging and DOT graph output.

        Returns
        -------
        logical_data
            New logical data with uninitialized values

        Examples
        --------
        >>> # Create uninitialized array (fast but contains garbage)
        >>> ld = ctx.logical_data_empty((100, 100), dtype=np.float32)

        >>> # Fast allocation without initialization
        >>> ld = ctx.logical_data_empty((50, 50, 50), name="tmp")
        """
        if dtype is None:
            dtype = np.float64
        return logical_data.init_by_shape(self, shape, dtype, name)

    def logical_data_full(self, shape, fill_value, dtype=None, where=None, exec_place=None, str name=None):
        """
        Create logical data initialized with a constant value.

        Similar to numpy.full(), this creates a new logical data with the given
        shape and fills it with fill_value.

        Parameters
        ----------
        shape : tuple
            Shape of the array
        fill_value : scalar
            Value to fill the array with
        dtype : numpy.dtype, optional
            Data type. If None, infer from fill_value.
        where : data_place, optional
            Data placement for initialization. Defaults to current device.
        exec_place : exec_place, optional
            Execution place for the fill operation. Defaults to current device.
            Note: exec_place.host() is not yet supported.
        name : str, optional
            Symbol name for debugging and DOT graph output.

        Returns
        -------
        logical_data
            New logical data initialized with fill_value

        Examples
        --------
        >>> # Create array filled with epsilon0 on current device
        >>> ld = ctx.logical_data_full((100, 100), 8.85e-12, dtype=np.float64)

        >>> # Create array on host memory
        >>> ld = ctx.logical_data_full((50, 50), 1.0, where=data_place.host())

        >>> # With a symbol name
        >>> ld = ctx.logical_data_full((200, 200), 0.0, name="epsilon")
        """
        # Infer dtype from fill_value if not provided
        if dtype is None:
            dtype = np.array(fill_value).dtype
        else:
            dtype = np.dtype(dtype)

        # Validate exec_place - host execution not yet supported
        if exec_place is not None:
            if hasattr(exec_place, 'kind') and exec_place.kind == "host":
                raise NotImplementedError(
                    "exec_place.host() is not yet supported for logical_data_full. "
                    "Use exec_place.device() or omit exec_place parameter."
                )

        # Create empty logical data
        ld = self.logical_data_empty(shape, dtype, name)

        # Initialize with the specified value (cuda.core.Buffer.fill; CuPy/Numba fallback for 8-byte)
        try:
            from cuda.stf.fill_utils import init_logical_data
            init_logical_data(self, ld, fill_value, where, exec_place)
        except ImportError as e:
            raise RuntimeError("Fill support (cuda.core) is not available for logical_data_full") from e

        return ld

    def logical_data_zeros(self, shape, dtype=None, where=None, exec_place=None, str name=None):
        """
        Create logical data filled with zeros.

        Equivalent to numpy.zeros() but for STF logical data.

        Parameters
        ----------
        shape : tuple
            Shape of the array
        dtype : numpy.dtype, optional
            Data type. Defaults to np.float64.
        where : data_place, optional
            Data placement. Defaults to current device.
        exec_place : exec_place, optional
            Execution place for the fill operation. Defaults to current device.
        name : str, optional
            Symbol name for debugging and DOT graph output.

        Returns
        -------
        logical_data
            New logical data filled with zeros

        Examples
        --------
        >>> # Create zero-filled array
        >>> ld = ctx.logical_data_zeros((100, 100), dtype=np.float32)

        >>> # Create on host memory with a name
        >>> ld = ctx.logical_data_zeros((50, 50), where=data_place.host(), name="Z")
        """
        if dtype is None:
            dtype = np.float64
        return self.logical_data_full(shape, 0.0, dtype, where, exec_place, name)

    def logical_data_ones(self, shape, dtype=None, where=None, exec_place=None, str name=None):
        """
        Create logical data filled with ones.

        Equivalent to numpy.ones() but for STF logical data.

        Parameters
        ----------
        shape : tuple
            Shape of the array
        dtype : numpy.dtype, optional
            Data type. Defaults to np.float64.
        where : data_place, optional
            Data placement. Defaults to current device.
        exec_place : exec_place, optional
            Execution place for the fill operation. Defaults to current device.
        name : str, optional
            Symbol name for debugging and DOT graph output.

        Returns
        -------
        logical_data
            New logical data filled with ones

        Examples
        --------
        >>> # Create ones-filled array
        >>> ld = ctx.logical_data_ones((100, 100), dtype=np.float32)

        >>> # Create on specific device with a name
        >>> ld = ctx.logical_data_ones((50, 50), name="ones")
        """
        if dtype is None:
            dtype = np.float64
        return self.logical_data_full(shape, 1.0, dtype, where, exec_place, name)

    def token(self):
        return logical_data.token(self)

    def task(self, *args, symbol=None):
        """
        Create a `task`

        Example
        -------
        >>> t = ctx.task(read(lX), rw(lY), symbol="axpy")
        >>> t.start()
        >>> t.end()
        """
        exec_place_set = False
        t = task(self)          # construct with this context
        if symbol is not None:
            t.set_symbol(symbol)
        for d in args:
            if isinstance(d, dep):
                t.add_dep(d)
            elif isinstance(d, exec_place):
                if exec_place_set:
                      raise ValueError("Only one exec_place can be given")
                t.set_exec_place(d)
                exec_place_set = True
            else:
                raise TypeError(
                    "Arguments must be dependency objects or an exec_place"
                )
        return t

    def cuda_kernel(self, *args, symbol=None):
        """Create an optimized CUDA kernel task.

        Accepts the same positional dep/exec_place arguments as
        ``ctx.task()``, but the resulting object exposes a ``launch()``
        method that describes a kernel to STF directly (enabling native
        graph-kernel nodes instead of stream capture).

        Example
        -------
        >>> with ctx.cuda_kernel(lX.read(), lY.rw(), symbol="axpy") as k:
        ...     dX, dY = k.get_arg(0), k.get_arg(1)
        ...     k.launch(kernel, grid=(4,), block=(256,),
        ...              args=[ctypes.c_int(N), ctypes.c_double(alpha), dX, dY])
        """
        exec_place_set = False
        k = cuda_kernel(self)
        if symbol is not None:
            k.set_symbol(symbol)
        for d in args:
            if isinstance(d, dep):
                k.add_dep(d)
            elif isinstance(d, exec_place):
                if exec_place_set:
                    raise ValueError("Only one exec_place can be given")
                k.set_exec_place(d)
                exec_place_set = True
            else:
                raise TypeError(
                    "Arguments must be dependency objects or an exec_place"
                )
        return k

    def host_launch(self, *deps, fn, args=None, symbol=None):
        """Schedule a host callback with dependency tracking.

        Deps (positional) are auto-unpacked as numpy arrays and passed as
        the first N arguments to ``fn``.  Extra user data goes through
        ``args`` and is appended after the dep arrays.

        Example::

            ctx.host_launch(lX.read(), fn=lambda x: print(x.sum()))
            ctx.host_launch(lX.read(), lY.read(), fn=check, args=[result])
        """
        if args is None:
            user_args = ()
        else:
            user_args = tuple(args)

        cdef logical_data ldata
        dep_meta = []
        for d in deps:
            if not isinstance(d, dep):
                raise TypeError(
                    "Positional arguments must be dep objects "
                    "(use ld.read(), ld.write(), or ld.rw())")
            ldata = <logical_data>d.ld
            dep_meta.append((ldata._shape, ldata._dtype))

        payload = (fn, user_args, dep_meta)
        Py_INCREF(payload)
        cdef PyObject* payload_ptr = <PyObject*>payload

        cdef stf_host_launch_handle h
        cdef int mode_ce
        h = stf_host_launch_create(self._ctx)
        if h == NULL:
            Py_XDECREF(<PyObject*>payload)
            raise RuntimeError("failed to create STF host_launch")
        try:
            if symbol is not None:
                sym_bytes = symbol.encode("utf-8")
                stf_host_launch_set_symbol(h, sym_bytes)
            for d in deps:
                ldata = <logical_data>d.ld
                mode_ce = <int>d.mode
                stf_host_launch_add_dep(h, ldata._ld, <stf_access_mode>mode_ce)
            stf_host_launch_set_user_data(
                h, &payload_ptr, sizeof(PyObject*), _python_payload_destructor)
            stf_host_launch_submit(h, _host_launch_trampoline)
        finally:
            stf_host_launch_destroy(h)
