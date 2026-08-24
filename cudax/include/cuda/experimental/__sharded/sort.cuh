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
 * @brief Sorting a sharded array: the two-tier design with two engines.
 *
 * The container tier (this function) owns placement, resources and
 * bookkeeping. The engine tier owns the cross-place choreography, and there
 * are two engines, one per rung of the cooperation-scope ladder:
 *
 *  - `sort_engine::shared_va` (`sort_shared_va.cuh`): when every shard lives
 *    in one shared address space (locality domains of one device, or the
 *    device itself), the combine uses what that rung shares — direct loads
 *    across shard boundaries. Local sorts, exact splitters by multi-sequence
 *    selection, and a fused gather-merge that writes each destination's slice
 *    straight into its own shard storage.
 *
 *  - `sort_engine::distributed`: the `__multi_gpu` MGMN distributed sort,
 *    programmed against the communicator concept and driven through
 *    `bind_engine` — tier 1 manufactures the communicators, environments and
 *    per-shard iterator/size ranges. The same construct that sorts across
 *    multi-process ranks sorts across in-process places here, unmodified:
 *    the portability path, and the only choice when the shards do not share
 *    an address space.
 *
 * Both engines honor the same contract: each shard ends holding the slice of
 * the globally sorted sequence at its ORIGINAL boundaries — sizes, offsets
 * and capacities unchanged by construction, and the fixed-boundary contract
 * of `allocate_contiguous` preserved (sorting a contiguous sharded array
 * leaves `contiguous_data()` reading as ONE globally sorted array). The
 * engine is therefore a performance choice, not an API change;
 * `sort_engine::automatic` picks by detection.
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
#include <cuda/experimental/__sharded/sort_shared_va.cuh>

#include <stdexcept>

namespace cuda::experimental::sharded
{
/**
 * @brief Tier-2 engine selection for `sharded::sort`.
 *
 * `automatic` uses the shared-address-space engine when every shard lives on
 * one device's address space (locality domains / the device itself) and the
 * distributed engine otherwise. The explicit values pin an engine — useful
 * for A/B comparison and for portability testing; requesting `shared_va`
 * where the shards do not share an address space throws.
 */
enum class sort_engine
{
  automatic, //!< detect: shared_va when the rung allows it, else distributed
  shared_va, //!< the in-process shared-address-space engine
  distributed //!< the MGMN communicator-based engine
};

/**
 * @brief Sort a sharded array globally, in place, with respect to @p comp.
 *
 * After the call, the concatenation of the shards (in shard order) is the
 * sorted permutation of the input, and each shard holds the slice of the
 * globally sorted sequence corresponding to its position — shard sizes,
 * offsets and capacities are unchanged. Works on every backing, including
 * contiguous (`allocate_contiguous`) arrays, whose fixed boundaries both
 * engines land on exactly. Sorting is not stable. Repeated runs on identical
 * input are bitwise identical (keys-only sort: the sorted multiset is
 * unique). SYNCHRONOUS: shards are sorted when this returns.
 *
 * @param group  the place group the array is sharded over (one shard per place)
 * @param data   the sharded array, sorted in place
 * @param comp   device-callable strict-weak-order comparator
 * @param engine tier-2 engine selection (default: detect)
 *
 * @throws std::invalid_argument when the array does not have one shard per
 *         group place, or when `sort_engine::shared_va` is requested but the
 *         shards do not share one device's address space.
 */
template <typename _Tp, typename _Compare = ::cuda::std::less<_Tp>>
void sort(place_group& group, sharded_array<_Tp>& data, _Compare comp = {}, sort_engine engine = sort_engine::automatic)
{
  if (data.size() == 0)
  {
    return;
  }

  check_places(data, group, "sharded::sort");

  // Both engines synchronize with the host (splitter/count readbacks) and
  // draw per-call temporaries: sorting cannot be recorded into a CUDA graph
  reserved::check_not_capturing(data, "sharded::sort");

  const bool va_eligible = reserved::one_shared_address_space(data);

  if (engine == sort_engine::shared_va && !va_eligible)
  {
    _CCCL_THROW(::std::invalid_argument,
                "sharded::sort: sort_engine::shared_va requires every shard on one device's address space");
  }

  if (engine == sort_engine::shared_va || (engine == sort_engine::automatic && va_eligible))
  {
    reserved::sort_shared_va(group, data, comp);
    return;
  }

  auto b = bind_engine(group, data);

  ::cuda::experimental::sort(::cuda::experimental::distributed, b.comms, b.envs, b.shard_data, b.shard_sizes, comp);

  // The engine redistributes each rank's slice back to its original count, so
  // the shard metadata is already correct; just drain the shard streams for
  // the synchronous contract shared with the other sharded collectives.
  data.sync();
}
} // namespace cuda::experimental::sharded
