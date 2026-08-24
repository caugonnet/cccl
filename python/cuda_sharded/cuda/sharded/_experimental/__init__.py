# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Experimental Python bindings for the ``cuda::experimental::sharded`` layer:
place-scoped sharded containers and the algorithms over them.

Containers bind the C++ implementation (one implementation of placement and
the fixed-size contract); tier-1 algorithms are one Python -> C++ crossing per
call; per-shard ``__cuda_array_interface__`` views are the interop escape
hatch. See the package README for the tier design and threading contract.
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("cuda-sharded")
except Exception:  # noqa: BLE001 - not installed (e.g. in-tree import)
    __version__ = "0.0.0"

# Importing the compiled bindings initializes the CUDA runtime lazily on first
# use, but the import itself loads CUDA libraries -- keep it lazy so that
# metadata-only imports of this package stay cheap.
_LAZY_SYMBOLS = {
    "place_group": "._sharded_bindings",
    "sharded_array": "._sharded_bindings",
    "shard_view": "._sharded_bindings",
    "fill": "._sharded_bindings",
    "sequence": "._sharded_bindings",
    "iota": "._sharded_bindings",
    "reduce": "._sharded_bindings",
    "inclusive_scan": "._sharded_bindings",
    "exclusive_scan": "._sharded_bindings",
    "adjacent_difference": "._sharded_bindings",
    "count": "._sharded_bindings",
    "histogram_even": "._sharded_bindings",
    "sort": "._sharded_bindings",
    "transform": "._sharded_bindings",
    "transform_binary": "._sharded_bindings",
    "SUPPORTED_DTYPES": "._sharded_bindings",
}

__all__ = [
    "SUPPORTED_DTYPES",
    "__version__",
    "adjacent_difference",
    "count",
    "exclusive_scan",
    "fill",
    "histogram_even",
    "inclusive_scan",
    "iota",
    "place_group",
    "reduce",
    "sequence",
    "shard_view",
    "sharded_array",
    "sort",
    "transform",
    "transform_binary",
]


def __getattr__(name):
    submodule = _LAZY_SYMBOLS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(submodule, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SYMBOLS))
