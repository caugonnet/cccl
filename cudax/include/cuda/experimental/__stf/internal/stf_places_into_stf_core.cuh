//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Pulls selected `cuda::experimental::places` names into `cuda::experimental::stf`.
 *
 * Include this **only from STF headers**, immediately after
 * `#include <cuda/experimental/__places/places.cuh>` (or equivalent), and before any
 * `namespace cuda::experimental::stf { ... }` that names those types unqualified.
 *
 * Places must not include STF for this bridge; dependency is STF → places only.
 */

#pragma once

namespace cuda::experimental::stf
{
using ::cuda::experimental::places::data_place;
using ::cuda::experimental::places::decorated_stream;
using ::cuda::experimental::places::device_ordinal;
using ::cuda::experimental::places::exec_place;
using ::cuda::experimental::places::exec_place_scope;
using ::cuda::experimental::places::from_index;
using ::cuda::experimental::places::partition_cyclic;
using ::cuda::experimental::places::partition_tile;
using ::cuda::experimental::places::to_index;
} // namespace cuda::experimental::stf
