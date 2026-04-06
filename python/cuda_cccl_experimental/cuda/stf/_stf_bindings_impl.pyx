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

    ctypedef int CUresult
    ctypedef OpaqueCUstream_st *CUstream
    ctypedef OpaqueCUkernel_st *CUkernel
    ctypedef OpaqueCUlibrary_st *CUlibrary

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

    #
    # Data places (opaque handles)
    #
    ctypedef struct stf_data_place_opaque_t
    ctypedef stf_data_place_opaque_t* stf_data_place_handle
    stf_data_place_handle stf_data_place_host()
    stf_data_place_handle stf_data_place_device(int dev_id)
    stf_data_place_handle stf_data_place_managed()
    stf_data_place_handle stf_data_place_affine()
    stf_data_place_handle stf_data_place_current_device()
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
    void* stf_task_get(stf_task_handle t, int submitted_index)
    void stf_task_destroy(stf_task_handle t)

    cdef enum stf_access_mode:
        STF_NONE
        STF_READ
        STF_WRITE
        STF_RW

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
            stf_logical_data_destroy(self._ld)
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
        out._ctx   = self._ctx
        out._dtype = self._dtype
        out._shape = self._shape
        out._ndim  = self._ndim
        out._len   = self._len
        out._symbol = None  # New object has no symbol initially
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
            stf_exec_place_destroy(self._h)
            self._h = NULL

    @staticmethod
    def device(int dev_id):
        cdef exec_place p = exec_place.__new__(exec_place)
        p._h = stf_exec_place_device(dev_id)
        return p

    @staticmethod
    def host():
        cdef exec_place p = exec_place.__new__(exec_place)
        p._h = stf_exec_place_host()
        return p

    @property
    def kind(self) -> str:
        if stf_exec_place_is_host(self._h):
            return "host"
        return "device"

cdef class data_place:
    cdef stf_data_place_handle _h

    def __cinit__(self):
        self._h = NULL

    def __dealloc__(self):
        if self._h != NULL:
            stf_data_place_destroy(self._h)
            self._h = NULL

    @staticmethod
    def device(int dev_id):
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_device(dev_id)
        return p

    @staticmethod
    def host():
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_host()
        return p

    @staticmethod
    def managed():
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_managed()
        return p

    @staticmethod
    def affine():
        cdef data_place p = data_place.__new__(data_place)
        p._h = stf_data_place_affine()
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
        self._lds_args = []

    def __dealloc__(self):
        if self._t != NULL:
             stf_task_destroy(self._t)

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
            self.finalize()

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

    def task(self, *args):
        """
        Create a `task`

        Example
        -------
        >>> t = ctx.task(read(lX), rw(lY))
        >>> t.start()
        >>> t.end()
        """
        exec_place_set = False
        t = task(self)          # construct with this context
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
