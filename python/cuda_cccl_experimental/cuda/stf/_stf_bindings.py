# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# _stf_bindings.py is a shim module that imports symbols from a
# _stf_bindings_impl extension module. The shim serves the same purposes as
# cuda.compute._bindings:
#
# 1. Import a CUDA-specific extension. The wheel ships cuda/stf/cu12/ and
#   cuda/stf/cu13/; at runtime this shim chooses based on the detected CUDA
#   version and imports all symbols from the matching extension.
#
# 2. Preload `nvrtc` and `nvJitLink` before importing the extension (indirect
#   dependencies via cccl.c.parallel / cccl.c.experimental.stf).

from __future__ import annotations

import importlib
import warnings

_BINDINGS_AVAILABLE = False

try:
    from cuda.cccl._cuda_version_utils import detect_cuda_version, get_recommended_extra
    from cuda.pathfinder import (  # type: ignore[import-not-found]
        load_nvidia_dynamic_lib,
    )
except ImportError as e:
    warnings.warn(
        f"CUDASTF dependencies not available: {e}. "
        "Install cuda-cccl-experimental[cu12] or cuda-cccl-experimental[cu13] "
        "to enable STF bindings.",
        RuntimeWarning,
    )
else:

    def _load_cuda_libraries():
        """Preload CUDA libraries to ensure proper symbol resolution."""
        for libname in ("nvrtc", "nvJitLink"):
            try:
                load_nvidia_dynamic_lib(libname)
            except Exception as exc:
                warnings.warn(
                    f"Failed to preload CUDA library '{libname}': {exc}. "
                    f"STF bindings may fail to load if {libname} is not available.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    _load_cuda_libraries()

    cuda_version = detect_cuda_version()
    if cuda_version not in [12, 13]:
        warnings.warn(
            f"Unsupported CUDA version: {cuda_version}. "
            "Only CUDA 12 and 13 are supported.",
            RuntimeWarning,
        )
    else:
        extra_name = get_recommended_extra(cuda_version)
        module_suffix = f".{extra_name}._stf_bindings_impl"

        try:
            bindings_module = importlib.import_module(module_suffix, __package__)
            globals().update(bindings_module.__dict__)
            _BINDINGS_AVAILABLE = True
        except ImportError as e:
            warnings.warn(
                f"CUDASTF bindings for CUDA {cuda_version} not available: {e}",
                RuntimeWarning,
            )
