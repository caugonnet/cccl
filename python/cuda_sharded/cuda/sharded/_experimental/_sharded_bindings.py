# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load the compiled bindings from the CUDA-major-specific directory.

The extension is installed into ``cu13/`` (or ``cu12/``) by CMake, mirroring
the layout of the sibling ``cuda-stf`` package; a source build ships exactly
one of them.
"""

from __future__ import annotations

import importlib

_BINDING_EXPORTS = (
    "place_group",
    "sharded_array",
    "shard_view",
    "fill",
    "sequence",
    "iota",
    "reduce",
    "inclusive_scan",
    "exclusive_scan",
    "adjacent_difference",
    "count",
    "histogram_even",
    "sort",
    "transform",
    "transform_binary",
    "SUPPORTED_DTYPES",
)


def _load():
    errors = []
    for extra in ("cu13", "cu12"):
        try:
            return importlib.import_module(
                f".{extra}._sharded_bindings_impl", __package__
            )
        except ImportError as e:
            errors.append(f"{extra}: {e}")
    raise ImportError(
        "could not import the cuda-sharded compiled bindings; tried:\n  "
        + "\n  ".join(errors)
    )


_impl = _load()

for _name in _BINDING_EXPORTS:
    globals()[_name] = getattr(_impl, _name)

__all__ = list(_BINDING_EXPORTS)
