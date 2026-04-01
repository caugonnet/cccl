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
 * @brief Additional `places` names and `stf::hash` specializations for place keys.
 *
 * Pulled after `stf::hash` is declared. Do not duplicate names from
 * `stf_places_into_stf_core.cuh`.
 */

#pragma once

#include <cuda/experimental/places.cuh>

namespace cuda::experimental::stf
{
using ::cuda::experimental::places::blocked_partition;
using ::cuda::experimental::places::cyclic_partition;
#if _CCCL_CTK_AT_LEAST(12, 4)
using ::cuda::experimental::places::green_context_helper;
using ::cuda::experimental::places::green_ctx_view;
#endif // _CCCL_CTK_AT_LEAST(12, 4)
using ::cuda::experimental::places::make_grid;
using ::cuda::experimental::places::partition_fn_t;
using ::cuda::experimental::places::place_partition;
using ::cuda::experimental::places::place_partition_scope;
using ::cuda::experimental::places::place_partition_scope_to_string;
using ::cuda::experimental::places::stream_pool;
using ::cuda::experimental::places::tiled;
using ::cuda::experimental::places::tiled_partition;

template <>
struct hash<::cuda::experimental::places::exec_place>
    : ::cuda::experimental::places::hash<::cuda::experimental::places::exec_place>
{};

template <>
struct hash<::cuda::experimental::places::data_place>
    : ::cuda::experimental::places::hash<::cuda::experimental::places::data_place>
{};
} // namespace cuda::experimental::stf
