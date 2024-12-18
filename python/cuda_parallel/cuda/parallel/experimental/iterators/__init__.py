from . import _iterators
import numba


def CacheModifiedInputIterator(device_array, modifier):
    """Python facade for Random Access Cache Modified Iterator that wraps a native device pointer.

    Similar to https://nvidia.github.io/cccl/cub/api/classcub_1_1CacheModifiedInputIterator.html

    Currently the only supported modifier is "stream" (LOAD_CS).
    """
    if modifier != "stream":
        raise NotImplementedError("Only stream modifier is supported")
    return _iterators.CacheModifiedPointer(
        device_array.__cuda_array_interface__["data"][0],
        numba.from_dtype(device_array.dtype),
    )


def ConstantIterator(value):
    """Python facade (similar to itertools.repeat) for C++ Random Access ConstantIterator."""
    return _iterators.ConstantIterator(value)


def CountingIterator(offset):
    """Python facade (similar to itertools.count) for C++ Random Access CountingIterator."""
    return _iterators.CountingIterator(offset)


def TransformIterator(op, it):
    """Python facade (similar to built-in map) mimicking a C++ Random Access TransformIterator."""
    return _iterators.make_transform_iterator(it, op)
