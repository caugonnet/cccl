//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Sorting a sharded array through the MGMN distributed sort as the
 *        engine.
 *
 * This is the two-tier structure at its clearest. The container tier (this
 * function) owns placement, resources and bookkeeping: it manufactures the
 * communicators, environments and per-shard iterator/size ranges via
 * `bind_engine`. The engine tier — the `__multi_gpu` distributed `sort` — owns
 * the cross-place choreography (sampling, histogramming, exchange, merge).
 * Because the engine is programmed against the communicator concept, the same
 * construct that sorts across multi-process ranks sorts across in-process
 * places here, unmodified.
 *
 * The engine's contract composes cleanly with the container's: it delivers to
 * each rank the slice of the globally sorted sequence belonging to that rank,
 * REDISTRIBUTED BACK TO THE RANK'S ORIGINAL ELEMENT COUNT. Shard sizes,
 * offsets and capacities are therefore unchanged by construction, and the
 * fixed-boundary contract of `allocate_contiguous` is preserved: sorting a
 * contiguous sharded array leaves `contiguous_data()` reading as ONE globally
 * sorted array.
 *
 * Engine swappability (the tier-2 seam): a future in-process specialization
 * (e.g. a placement-aware multi-place radix sort) can replace the engine
 * behind this same name and contract — a performance change, not an API
 * change.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/functional>

#include <cuda/experimental/__multi_gpu/algorithm/sort/sort.h>
#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/communicator.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

namespace cuda::experimental::sharded
{
/**
 * @brief Sort a sharded array globally, in place, with respect to @p comp.
 *
 * After the call, the concatenation of the shards (in shard order) is the
 * sorted permutation of the input, and each shard holds the slice of the
 * globally sorted sequence corresponding to its position — shard sizes,
 * offsets and capacities are unchanged. Works on every backing, including
 * contiguous (`allocate_contiguous`) arrays, whose fixed boundaries the
 * engine's redistribution lands on exactly. Sorting is not stable.
 *
 * Tier 1 (container): communicators, environments (stream + per-place memory
 * resource, so engine temporaries land on the place whose rank uses them),
 * iterators and sizes come from `bind_engine`. Tier 2 (engine): the MGMN
 * distributed sort over those ranges. SYNCHRONOUS: shards are sorted when
 * this returns.
 *
 * @param group the place group the array is sharded over (one shard per place)
 * @param data  the sharded array, sorted in place
 * @param comp  device-callable strict-weak-order comparator
 *
 * @throws std::invalid_argument when the array does not have one shard per
 *         group place (via `bind_engine`).
 */
template <typename _Tp, typename _Compare = ::cuda::std::less<_Tp>>
void sort(place_group& group, sharded_array<_Tp>& data, _Compare comp = {})
{
  if (data.size() == 0)
  {
    return;
  }

  check_places(data, group, "sharded::sort");

  auto b = bind_engine(group, data);

  ::cuda::experimental::sort(::cuda::experimental::distributed, b.comms, b.envs, b.shard_data, b.shard_sizes, comp);

  // The engine redistributes each rank's slice back to its original count, so
  // the shard metadata is already correct; just drain the shard streams for
  // the synchronous contract shared with the other sharded collectives.
  data.sync();
}
} // namespace cuda::experimental::sharded
