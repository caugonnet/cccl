/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_segmented_radix_sort.cuh>
#include <cub/agent/agent_sub_warp_merge_sort.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/detail/device_double_buffer.cuh>
#include <cub/detail/temporary_storage.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort.cuh>
#include <cub/thread/thread_sort.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/cmath>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/type_traits>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort
{
// Type used to index within segments within a single invocation
using local_segment_index_t = ::cuda::std::uint32_t;
// Type used for total number of segments and to index within segments globally
using global_segment_offset_t = ::cuda::std::int64_t;

/**
 * @brief Fallback kernel, in case there's not enough segments to
 *        take advantage of partitioning.
 *
 * In this case, the sorting method is still selected based on the segment size.
 * If a single warp can sort the segment, the algorithm will use the sub-warp
 * merge sort. Otherwise, the algorithm will use the in-shared-memory version of
 * block radix sort. If data don't fit into shared memory, the algorithm will
 * use in-global-memory radix sort.
 *
 * @param[in] d_keys_in_orig
 *   Input keys buffer
 *
 * @param[out] d_keys_out_orig
 *   Output keys buffer
 *
 * @param[in,out] d_keys_double_buffer
 *   Double keys buffer
 *
 * @param[in] d_values_in_orig
 *   Input values buffer
 *
 * @param[out] d_values_out_orig
 *   Output values buffer
 *
 * @param[in,out] d_values_double_buffer
 *   Double values buffer
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length
 *   `num_segments`, such that `d_begin_offsets[i]` is the first element of the
 *   i-th data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length
 *   `num_segments`, such that `d_end_offsets[i]-1` is the last element of the
 *   i-th data segment in `d_keys_*` and `d_values_*`.
 *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
 *   considered empty.
 */
template <SortOrder Order,
          typename ChainedPolicyT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::LargeSegmentPolicy::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedSortFallbackKernel(
    const KeyT* d_keys_in_orig,
    KeyT* d_keys_out_orig,
    device_double_buffer<KeyT> d_keys_double_buffer,
    const ValueT* d_values_in_orig,
    ValueT* d_values_out_orig,
    device_double_buffer<ValueT> d_values_double_buffer,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets)
{
  using ActivePolicyT       = typename ChainedPolicyT::ActivePolicy;
  using LargeSegmentPolicyT = typename ActivePolicyT::LargeSegmentPolicy;
  using MediumPolicyT       = typename ActivePolicyT::SmallAndMediumSegmentedSortPolicyT::MediumPolicyT;

  const auto segment_id = static_cast<local_segment_index_t>(blockIdx.x);
  OffsetT segment_begin = d_begin_offsets[segment_id];
  OffsetT segment_end   = d_end_offsets[segment_id];
  OffsetT num_items     = segment_end - segment_begin;

  if (num_items <= 0)
  {
    return;
  }

  using AgentSegmentedRadixSortT =
    radix_sort::AgentSegmentedRadixSort<Order == SortOrder::Descending, LargeSegmentPolicyT, KeyT, ValueT, OffsetT>;

  using WarpReduceT = cub::WarpReduce<KeyT>;

  using AgentWarpMergeSortT =
    sub_warp_merge_sort::AgentSubWarpSort<Order == SortOrder::Descending, MediumPolicyT, KeyT, ValueT, OffsetT>;

  __shared__ union
  {
    typename AgentSegmentedRadixSortT::TempStorage block_sort;
    typename WarpReduceT::TempStorage warp_reduce;
    typename AgentWarpMergeSortT::TempStorage medium_warp_sort;
  } temp_storage;

  constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;
  AgentSegmentedRadixSortT agent(num_items, temp_storage.block_sort);

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(KeyT) * 8;

  constexpr int cacheable_tile_size = LargeSegmentPolicyT::BLOCK_THREADS * LargeSegmentPolicyT::ITEMS_PER_THREAD;

  d_keys_in_orig += segment_begin;
  d_keys_out_orig += segment_begin;

  if (!keys_only)
  {
    d_values_in_orig += segment_begin;
    d_values_out_orig += segment_begin;
  }

  if (num_items <= MediumPolicyT::ITEMS_PER_TILE)
  {
    // Sort by a single warp
    if (threadIdx.x < MediumPolicyT::WARP_THREADS)
    {
      AgentWarpMergeSortT(temp_storage.medium_warp_sort)
        .ProcessSegment(num_items, d_keys_in_orig, d_keys_out_orig, d_values_in_orig, d_values_out_orig);
    }
  }
  else if (num_items < cacheable_tile_size)
  {
    // Sort by a CTA if data fits into shared memory
    agent.ProcessSinglePass(begin_bit, end_bit, d_keys_in_orig, d_values_in_orig, d_keys_out_orig, d_values_out_orig);
  }
  else
  {
    // Sort by a CTA with multiple reads from global memory
    int current_bit = begin_bit;
    int pass_bits   = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

    d_keys_double_buffer = device_double_buffer<KeyT>(
      d_keys_double_buffer.current() + segment_begin, d_keys_double_buffer.alternate() + segment_begin);

    if (!keys_only)
    {
      d_values_double_buffer = device_double_buffer<ValueT>(
        d_values_double_buffer.current() + segment_begin, d_values_double_buffer.alternate() + segment_begin);
    }

    agent.ProcessIterative(
      current_bit,
      pass_bits,
      d_keys_in_orig,
      d_values_in_orig,
      d_keys_double_buffer.current(),
      d_values_double_buffer.current());
    current_bit += pass_bits;

    _CCCL_PRAGMA_NOUNROLL()
    while (current_bit < end_bit)
    {
      pass_bits = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

      __syncthreads();
      agent.ProcessIterative(
        current_bit,
        pass_bits,
        d_keys_double_buffer.current(),
        d_values_double_buffer.current(),
        d_keys_double_buffer.alternate(),
        d_values_double_buffer.alternate());

      d_keys_double_buffer.swap();
      d_values_double_buffer.swap();
      current_bit += pass_bits;
    }
  }
}

/**
 * @brief Single kernel for moderate size (less than a few thousand items)
 *        segments.
 *
 * This kernel allocates a sub-warp per segment. Therefore, this kernel assigns
 * a single thread block to multiple segments. Segments fall into two
 * categories. An architectural warp usually sorts segments in the medium-size
 * category, while a few threads sort segments in the small-size category. Since
 * segments are partitioned, we know the last thread block index assigned to
 * sort medium-size segments. A particular thread block can check this number to
 * find out which category it was assigned to sort. In both cases, the
 * merge sort is used.
 *
 * @param[in] small_segments
 *   Number of segments that can be sorted by a warp part
 *
 * @param[in] medium_segments
 *   Number of segments that can be sorted by a warp
 *
 * @param[in] medium_blocks
 *   Number of CTAs assigned to process medium segments
 *
 * @param[in] d_small_segments_indices
 *   Small segments mapping of length @p small_segments, such that
 *   `d_small_segments_indices[i]` is the input segment index
 *
 * @param[in] d_medium_segments_indices
 *   Medium segments mapping of length @p medium_segments, such that
 *   `d_medium_segments_indices[i]` is the input segment index
 *
 * @param[in] d_keys_in_orig
 *   Input keys buffer
 *
 * @param[out] d_keys_out_orig
 *   Output keys buffer
 *
 * @param[in] d_values_in_orig
 *   Input values buffer
 *
 * @param[out] d_values_out_orig
 *   Output values buffer
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length
 *   `num_segments`, such that `d_begin_offsets[i]` is the first element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length
 *   `num_segments`, such that `d_end_offsets[i]-1` is the last element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`. If
 *   `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the <em>i</em><sup>th</sup> is
 *   considered empty.
 */
template <SortOrder Order,
          typename ChainedPolicyT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::SmallAndMediumSegmentedSortPolicyT::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedSortKernelSmall(
    local_segment_index_t small_segments,
    local_segment_index_t medium_segments,
    local_segment_index_t medium_blocks,
    const local_segment_index_t* d_small_segments_indices,
    const local_segment_index_t* d_medium_segments_indices,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets)
{
  using local_segment_index_t = local_segment_index_t;

  const local_segment_index_t tid = threadIdx.x;
  const local_segment_index_t bid = blockIdx.x;

  using ActivePolicyT         = typename ChainedPolicyT::ActivePolicy;
  using SmallAndMediumPolicyT = typename ActivePolicyT::SmallAndMediumSegmentedSortPolicyT;
  using MediumPolicyT         = typename SmallAndMediumPolicyT::MediumPolicyT;
  using SmallPolicyT          = typename SmallAndMediumPolicyT::SmallPolicyT;

  constexpr auto threads_per_medium_segment = static_cast<local_segment_index_t>(MediumPolicyT::WARP_THREADS);
  constexpr auto threads_per_small_segment  = static_cast<local_segment_index_t>(SmallPolicyT::WARP_THREADS);

  using MediumAgentWarpMergeSortT =
    sub_warp_merge_sort::AgentSubWarpSort<Order == SortOrder::Descending, MediumPolicyT, KeyT, ValueT, OffsetT>;

  using SmallAgentWarpMergeSortT =
    sub_warp_merge_sort::AgentSubWarpSort<Order == SortOrder::Descending, SmallPolicyT, KeyT, ValueT, OffsetT>;

  constexpr auto segments_per_medium_block =
    static_cast<local_segment_index_t>(SmallAndMediumPolicyT::SEGMENTS_PER_MEDIUM_BLOCK);

  constexpr auto segments_per_small_block =
    static_cast<local_segment_index_t>(SmallAndMediumPolicyT::SEGMENTS_PER_SMALL_BLOCK);

  __shared__ union
  {
    typename MediumAgentWarpMergeSortT::TempStorage medium_storage[segments_per_medium_block];

    typename SmallAgentWarpMergeSortT::TempStorage small_storage[segments_per_small_block];
  } temp_storage;

  if (bid < medium_blocks)
  {
    const local_segment_index_t sid_within_block  = tid / threads_per_medium_segment;
    const local_segment_index_t medium_segment_id = bid * segments_per_medium_block + sid_within_block;

    if (medium_segment_id < medium_segments)
    {
      const local_segment_index_t global_segment_id = d_medium_segments_indices[medium_segment_id];

      const OffsetT segment_begin = d_begin_offsets[global_segment_id];
      const OffsetT segment_end   = d_end_offsets[global_segment_id];
      const OffsetT num_items     = segment_end - segment_begin;

      MediumAgentWarpMergeSortT(temp_storage.medium_storage[sid_within_block])
        .ProcessSegment(num_items,
                        d_keys_in + segment_begin,
                        d_keys_out + segment_begin,
                        d_values_in + segment_begin,
                        d_values_out + segment_begin);
    }
  }
  else
  {
    const local_segment_index_t sid_within_block = tid / threads_per_small_segment;
    const local_segment_index_t small_segment_id = (bid - medium_blocks) * segments_per_small_block + sid_within_block;

    if (small_segment_id < small_segments)
    {
      const local_segment_index_t global_segment_id = d_small_segments_indices[small_segment_id];

      const OffsetT segment_begin = d_begin_offsets[global_segment_id];
      const OffsetT segment_end   = d_end_offsets[global_segment_id];
      const OffsetT num_items     = segment_end - segment_begin;

      SmallAgentWarpMergeSortT(temp_storage.small_storage[sid_within_block])
        .ProcessSegment(num_items,
                        d_keys_in + segment_begin,
                        d_keys_out + segment_begin,
                        d_values_in + segment_begin,
                        d_values_out + segment_begin);
    }
  }
}

/**
 * @brief Single kernel for large size (more than a few thousand items) segments.
 *
 * @param[in] d_keys_in_orig
 *   Input keys buffer
 *
 * @param[out] d_keys_out_orig
 *   Output keys buffer
 *
 * @param[in] d_values_in_orig
 *   Input values buffer
 *
 * @param[out] d_values_out_orig
 *   Output values buffer
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length
 *   `num_segments`, such that `d_begin_offsets[i]` is the first element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length
 *   `num_segments`, such that `d_end_offsets[i]-1` is the last element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`. If
 *   `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the <em>i</em><sup>th</sup> is
 *   considered empty.
 */
template <SortOrder Order,
          typename ChainedPolicyT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::LargeSegmentPolicy::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedSortKernelLarge(
    const local_segment_index_t* d_segments_indices,
    const KeyT* d_keys_in_orig,
    KeyT* d_keys_out_orig,
    device_double_buffer<KeyT> d_keys_double_buffer,
    const ValueT* d_values_in_orig,
    ValueT* d_values_out_orig,
    device_double_buffer<ValueT> d_values_double_buffer,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets)
{
  using ActivePolicyT         = typename ChainedPolicyT::ActivePolicy;
  using LargeSegmentPolicyT   = typename ActivePolicyT::LargeSegmentPolicy;
  using local_segment_index_t = local_segment_index_t;

  constexpr int small_tile_size = LargeSegmentPolicyT::BLOCK_THREADS * LargeSegmentPolicyT::ITEMS_PER_THREAD;

  using AgentSegmentedRadixSortT =
    radix_sort::AgentSegmentedRadixSort<Order == SortOrder::Descending, LargeSegmentPolicyT, KeyT, ValueT, OffsetT>;

  __shared__ typename AgentSegmentedRadixSortT::TempStorage storage;

  const local_segment_index_t bid = blockIdx.x;

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(KeyT) * 8;

  const local_segment_index_t global_segment_id = d_segments_indices[bid];
  const OffsetT segment_begin                   = d_begin_offsets[global_segment_id];
  const OffsetT segment_end                     = d_end_offsets[global_segment_id];
  const OffsetT num_items                       = segment_end - segment_begin;

  constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;
  AgentSegmentedRadixSortT agent(num_items, storage);

  d_keys_in_orig += segment_begin;
  d_keys_out_orig += segment_begin;

  if (!keys_only)
  {
    d_values_in_orig += segment_begin;
    d_values_out_orig += segment_begin;
  }

  if (num_items < small_tile_size)
  {
    // Sort in shared memory if the segment fits into it
    agent.ProcessSinglePass(begin_bit, end_bit, d_keys_in_orig, d_values_in_orig, d_keys_out_orig, d_values_out_orig);
  }
  else
  {
    // Sort reading global memory multiple times
    int current_bit = begin_bit;
    int pass_bits   = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

    d_keys_double_buffer = device_double_buffer<KeyT>(
      d_keys_double_buffer.current() + segment_begin, d_keys_double_buffer.alternate() + segment_begin);

    if (!keys_only)
    {
      d_values_double_buffer = device_double_buffer<ValueT>(
        d_values_double_buffer.current() + segment_begin, d_values_double_buffer.alternate() + segment_begin);
    }

    agent.ProcessIterative(
      current_bit,
      pass_bits,
      d_keys_in_orig,
      d_values_in_orig,
      d_keys_double_buffer.current(),
      d_values_double_buffer.current());
    current_bit += pass_bits;

    _CCCL_PRAGMA_NOUNROLL()
    while (current_bit < end_bit)
    {
      pass_bits = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

      __syncthreads();
      agent.ProcessIterative(
        current_bit,
        pass_bits,
        d_keys_double_buffer.current(),
        d_values_double_buffer.current(),
        d_keys_double_buffer.alternate(),
        d_values_double_buffer.alternate());

      d_keys_double_buffer.swap();
      d_values_double_buffer.swap();
      current_bit += pass_bits;
    }
  }
}

/*
 * Continuation is called after the partitioning stage. It launches kernels
 * to sort large and small segments using the partitioning results. Separation
 * of this stage is required to eliminate device-side synchronization in
 * the CDP mode.
 */
template <typename LargeSegmentPolicyT,
          typename SmallAndMediumPolicyT,
          typename LargeKernelT,
          typename SmallKernelT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN cudaError_t DeviceSegmentedSortContinuation(
  LargeKernelT large_kernel,
  SmallKernelT small_kernel,
  int num_segments,
  KeyT* d_current_keys,
  KeyT* d_final_keys,
  device_double_buffer<KeyT> d_keys_double_buffer,
  ValueT* d_current_values,
  ValueT* d_final_values,
  device_double_buffer<ValueT> d_values_double_buffer,
  BeginOffsetIteratorT d_begin_offsets,
  EndOffsetIteratorT d_end_offsets,
  local_segment_index_t* group_sizes,
  local_segment_index_t* large_and_medium_segments_indices,
  local_segment_index_t* small_segments_indices,
  cudaStream_t stream)
{
  using local_segment_index_t = local_segment_index_t;

  cudaError error = cudaSuccess;

  const local_segment_index_t large_segments = group_sizes[0];

  if (large_segments > 0)
  {
    // One CTA per segment
    const local_segment_index_t blocks_in_grid = large_segments;

#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking "
            "DeviceSegmentedSortKernelLarge<<<%d, %d, 0, %lld>>>()\n",
            static_cast<int>(blocks_in_grid),
            LargeSegmentPolicyT::BLOCK_THREADS,
            (long long) stream);
#endif // CUB_DEBUG_LOG

    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(blocks_in_grid, LargeSegmentPolicyT::BLOCK_THREADS, 0, stream)
      .doit(large_kernel,
            large_and_medium_segments_indices,
            d_current_keys,
            d_final_keys,
            d_keys_double_buffer,
            d_current_values,
            d_final_values,
            d_values_double_buffer,
            d_begin_offsets,
            d_end_offsets);

    // Check for failure to launch
    error = CubDebug(cudaPeekAtLastError());
    if (cudaSuccess != error)
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    error = CubDebug(DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      return error;
    }
  }

  const local_segment_index_t small_segments = group_sizes[1];
  const local_segment_index_t medium_segments =
    static_cast<local_segment_index_t>(num_segments) - (large_segments + small_segments);

  const local_segment_index_t small_blocks =
    ::cuda::ceil_div(small_segments, SmallAndMediumPolicyT::SEGMENTS_PER_SMALL_BLOCK);

  const local_segment_index_t medium_blocks =
    ::cuda::ceil_div(medium_segments, SmallAndMediumPolicyT::SEGMENTS_PER_MEDIUM_BLOCK);

  const local_segment_index_t small_and_medium_blocks_in_grid = small_blocks + medium_blocks;

  if (small_and_medium_blocks_in_grid)
  {
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking "
            "DeviceSegmentedSortKernelSmall<<<%d, %d, 0, %lld>>>()\n",
            static_cast<int>(small_and_medium_blocks_in_grid),
            SmallAndMediumPolicyT::BLOCK_THREADS,
            (long long) stream);
#endif // CUB_DEBUG_LOG

    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
      small_and_medium_blocks_in_grid, SmallAndMediumPolicyT::BLOCK_THREADS, 0, stream)
      .doit(small_kernel,
            small_segments,
            medium_segments,
            medium_blocks,
            small_segments_indices,
            large_and_medium_segments_indices + num_segments - medium_segments,
            d_current_keys,
            d_final_keys,
            d_current_values,
            d_final_values,
            d_begin_offsets,
            d_end_offsets);

    // Check for failure to launch
    error = CubDebug(cudaPeekAtLastError());
    if (cudaSuccess != error)
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    error = CubDebug(DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      return error;
    }
  }

  return error;
}

#ifdef CUB_RDC_ENABLED
/*
 * Continuation kernel is used only in the CDP mode. It's used to
 * launch DeviceSegmentedSortContinuation as a separate kernel.
 */
template <typename ChainedPolicyT,
          typename LargeKernelT,
          typename SmallKernelT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT>
__launch_bounds__(1) CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedSortContinuationKernel(
  LargeKernelT large_kernel,
  SmallKernelT small_kernel,
  local_segment_index_t num_segments,
  KeyT* d_current_keys,
  KeyT* d_final_keys,
  device_double_buffer<KeyT> d_keys_double_buffer,
  ValueT* d_current_values,
  ValueT* d_final_values,
  device_double_buffer<ValueT> d_values_double_buffer,
  BeginOffsetIteratorT d_begin_offsets,
  EndOffsetIteratorT d_end_offsets,
  local_segment_index_t* group_sizes,
  local_segment_index_t* large_and_medium_segments_indices,
  local_segment_index_t* small_segments_indices)
{
  using ActivePolicyT         = typename ChainedPolicyT::ActivePolicy;
  using LargeSegmentPolicyT   = typename ActivePolicyT::LargeSegmentPolicy;
  using SmallAndMediumPolicyT = typename ActivePolicyT::SmallAndMediumSegmentedSortPolicyT;

  // In case of CDP:
  // 1. each CTA has a different main stream
  // 2. all streams are non-blocking
  // 3. child grid always completes before the parent grid
  // 4. streams can be used only from the CTA in which they were created
  // 5. streams created on the host cannot be used on the device
  //
  // Due to (4, 5), we can't pass the user-provided stream in the continuation.
  // Due to (1, 2, 3) it's safe to pass the main stream.
  cudaError_t error =
    detail::segmented_sort::DeviceSegmentedSortContinuation<LargeSegmentPolicyT, SmallAndMediumPolicyT>(
      large_kernel,
      small_kernel,
      num_segments,
      d_current_keys,
      d_final_keys,
      d_keys_double_buffer,
      d_current_values,
      d_final_values,
      d_values_double_buffer,
      d_begin_offsets,
      d_end_offsets,
      group_sizes,
      large_and_medium_segments_indices,
      small_segments_indices,
      0); // always launching on the main stream (see motivation above)

  error = CubDebug(error);
}
#endif // CUB_RDC_ENABLED

} // namespace detail::segmented_sort

template <SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename PolicyHub = detail::segmented_sort::policy_hub<KeyT, ValueT>>
struct DispatchSegmentedSort
{
  using local_segment_index_t   = detail::segmented_sort::local_segment_index_t;
  using global_segment_offset_t = detail::segmented_sort::global_segment_offset_t;

  static constexpr int KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  struct LargeSegmentsSelectorT
  {
    OffsetT value{};
    BeginOffsetIteratorT d_offset_begin{};
    EndOffsetIteratorT d_offset_end{};
    global_segment_offset_t base_segment_offset{};

    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
    LargeSegmentsSelectorT(OffsetT value, BeginOffsetIteratorT d_offset_begin, EndOffsetIteratorT d_offset_end)
        : value(value)
        , d_offset_begin(d_offset_begin)
        , d_offset_end(d_offset_end)
    {}

    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator()(local_segment_index_t segment_id) const
    {
      const OffsetT segment_size =
        d_offset_end[base_segment_offset + segment_id] - d_offset_begin[base_segment_offset + segment_id];
      return segment_size > value;
    }
  };

  struct SmallSegmentsSelectorT
  {
    OffsetT value{};
    BeginOffsetIteratorT d_offset_begin{};
    EndOffsetIteratorT d_offset_end{};
    global_segment_offset_t base_segment_offset{};

    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
    SmallSegmentsSelectorT(OffsetT value, BeginOffsetIteratorT d_offset_begin, EndOffsetIteratorT d_offset_end)
        : value(value)
        , d_offset_begin(d_offset_begin)
        , d_offset_end(d_offset_end)
    {}

    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator()(local_segment_index_t segment_id) const
    {
      const OffsetT segment_size =
        d_offset_end[base_segment_offset + segment_id] - d_offset_begin[base_segment_offset + segment_id];
      return segment_size < value;
    }
  };

  // Partition selects large and small groups. The middle group is not selected.
  static constexpr size_t num_selected_groups = 2;

  /**
   * Device-accessible allocation of temporary storage. When `nullptr`, the
   * required allocation size is written to `temp_storage_bytes` and no work
   * is done.
   */
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /**
   * Double-buffer whose current buffer contains the unsorted input keys and,
   * upon return, is updated to point to the sorted output keys
   */
  DoubleBuffer<KeyT>& d_keys;

  /**
   * Double-buffer whose current buffer contains the unsorted input values and,
   * upon return, is updated to point to the sorted output values
   */
  DoubleBuffer<ValueT>& d_values;

  /// Number of items to sort
  ::cuda::std::int64_t num_items;

  /// The number of segments that comprise the sorting data
  global_segment_offset_t num_segments;

  /**
   * Random-access input iterator to the sequence of beginning offsets of length
   * `num_segments`, such that `d_begin_offsets[i]` is the first element of the
   * <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`
   */
  BeginOffsetIteratorT d_begin_offsets;

  /**
   * Random-access input iterator to the sequence of ending offsets of length
   * `num_segments`, such that <tt>d_end_offsets[i]-1</tt> is the last element
   * of the <em>i</em><sup>th</sup> data segment in `d_keys_*` and
   * `d_values_*`. If `d_end_offsets[i]-1 <= d_begin_offsets[i]`,
   * the <em>i</em><sup>th</sup> is considered empty.
   */
  EndOffsetIteratorT d_end_offsets;

  /// Whether is okay to overwrite source buffers
  bool is_overwrite_okay;

  /// CUDA stream to launch kernels within.
  cudaStream_t stream;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchSegmentedSort(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    ::cuda::std::int64_t num_items,
    global_segment_offset_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    bool is_overwrite_okay,
    cudaStream_t stream)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys(d_keys)
      , d_values(d_values)
      , num_items(num_items)
      , num_segments(num_segments)
      , d_begin_offsets(d_begin_offsets)
      , d_end_offsets(d_end_offsets)
      , is_overwrite_okay(is_overwrite_okay)
      , stream(stream)
  {}

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using LargeSegmentPolicyT   = typename ActivePolicyT::LargeSegmentPolicy;
    using SmallAndMediumPolicyT = typename ActivePolicyT::SmallAndMediumSegmentedSortPolicyT;

    static_assert(LargeSegmentPolicyT::LOAD_MODIFIER != CacheLoadModifier::LOAD_LDG,
                  "The memory consistency model does not apply to texture accesses");

    static_assert(KEYS_ONLY || LargeSegmentPolicyT::LOAD_ALGORITHM != BLOCK_LOAD_STRIPED
                    || SmallAndMediumPolicyT::MediumPolicyT::LOAD_ALGORITHM != WARP_LOAD_STRIPED
                    || SmallAndMediumPolicyT::SmallPolicyT::LOAD_ALGORITHM != WARP_LOAD_STRIPED,
                  "Striped load will make this algorithm unstable");

    static_assert(SmallAndMediumPolicyT::MediumPolicyT::STORE_ALGORITHM != WARP_STORE_STRIPED
                    || SmallAndMediumPolicyT::SmallPolicyT::STORE_ALGORITHM != WARP_STORE_STRIPED,
                  "Striped stores will produce unsorted results");

    constexpr int radix_bits = LargeSegmentPolicyT::RADIX_BITS;

    cudaError error = cudaSuccess;

    do
    {
      //------------------------------------------------------------------------
      // Prepare temporary storage layout
      //------------------------------------------------------------------------

      const bool partition_segments = num_segments > ActivePolicyT::PARTITIONING_THRESHOLD;

      cub::detail::temporary_storage::layout<5> temporary_storage_layout;

      auto keys_slot                          = temporary_storage_layout.get_slot(0);
      auto values_slot                        = temporary_storage_layout.get_slot(1);
      auto large_and_medium_partitioning_slot = temporary_storage_layout.get_slot(2);
      auto small_partitioning_slot            = temporary_storage_layout.get_slot(3);
      auto group_sizes_slot                   = temporary_storage_layout.get_slot(4);

      auto keys_allocation   = keys_slot->create_alias<KeyT>();
      auto values_allocation = values_slot->create_alias<ValueT>();

      if (!is_overwrite_okay)
      {
        keys_allocation.grow(num_items);

        if (!KEYS_ONLY)
        {
          values_allocation.grow(num_items);
        }
      }

      auto large_and_medium_segments_indices =
        large_and_medium_partitioning_slot->create_alias<local_segment_index_t>();
      auto small_segments_indices = small_partitioning_slot->create_alias<local_segment_index_t>();
      auto group_sizes            = group_sizes_slot->create_alias<local_segment_index_t>();

      size_t three_way_partition_temp_storage_bytes{};

      LargeSegmentsSelectorT large_segments_selector(
        SmallAndMediumPolicyT::MediumPolicyT::ITEMS_PER_TILE, d_begin_offsets, d_end_offsets);

      SmallSegmentsSelectorT small_segments_selector(
        SmallAndMediumPolicyT::SmallPolicyT::ITEMS_PER_TILE + 1, d_begin_offsets, d_end_offsets);

      auto device_partition_temp_storage = keys_slot->create_alias<uint8_t>();

      if (partition_segments)
      {
        constexpr auto num_segments_per_invocation_limit =
          static_cast<global_segment_offset_t>(::cuda::std::numeric_limits<int>::max());
        auto const max_num_segments_per_invocation = static_cast<global_segment_offset_t>(
          (::cuda::std::min) (static_cast<global_segment_offset_t>(num_segments), num_segments_per_invocation_limit));

        large_and_medium_segments_indices.grow(max_num_segments_per_invocation);
        small_segments_indices.grow(max_num_segments_per_invocation);
        group_sizes.grow(num_selected_groups);

        auto medium_indices_iterator =
          THRUST_NS_QUALIFIER::make_reverse_iterator(large_and_medium_segments_indices.get());

        cub::DevicePartition::IfNoNVTX(
          nullptr,
          three_way_partition_temp_storage_bytes,
          THRUST_NS_QUALIFIER::counting_iterator<local_segment_index_t>(0),
          large_and_medium_segments_indices.get(),
          small_segments_indices.get(),
          medium_indices_iterator,
          group_sizes.get(),
          max_num_segments_per_invocation,
          large_segments_selector,
          small_segments_selector,
          stream);

        device_partition_temp_storage.grow(three_way_partition_temp_storage_bytes);
      }

      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = temporary_storage_layout.get_size();

        // Return if the caller is simply requesting the size of the storage
        // allocation
        break;
      }

      if (num_items == 0 || num_segments == 0)
      {
        break;
      }

      error = CubDebug(temporary_storage_layout.map_to_buffer(d_temp_storage, temp_storage_bytes));
      if (cudaSuccess != error)
      {
        break;
      }

      //------------------------------------------------------------------------
      // Sort
      //------------------------------------------------------------------------

      const bool is_num_passes_odd = GetNumPasses(radix_bits) & 1;

      /**
       * This algorithm sorts segments that don't fit into shared memory with
       * the in-global-memory radix sort. Radix sort splits key representation
       * into multiple "digits". Each digit is RADIX_BITS wide. The algorithm
       * iterates over these digits. Each of these iterations consists of a
       * couple of stages. The first stage computes a histogram for a current
       * digit in each segment key. This histogram helps to determine the
       * starting position of the keys group with a similar digit.
       * For example:
       * keys_digits  = [ 1, 0, 0, 1 ]
       * digit_prefix = [ 0, 2 ]
       * The second stage checks the keys again and increments the prefix to
       * determine the final position of the key:
       *
       *               expression            |  key  |   idx   |     result
       * ----------------------------------- | ----- | ------- | --------------
       * result[prefix[keys[0]]++] = keys[0] |   1   |    2    | [ ?, ?, 1, ? ]
       * result[prefix[keys[1]]++] = keys[0] |   0   |    0    | [ 0, ?, 1, ? ]
       * result[prefix[keys[2]]++] = keys[0] |   0   |    1    | [ 0, 0, 1, ? ]
       * result[prefix[keys[3]]++] = keys[0] |   1   |    3    | [ 0, 0, 1, 1 ]
       *
       * If the resulting memory is aliased to the input one, we'll face the
       * following issues:
       *
       *     input      |  key  |   idx   |   result/input   |      issue
       * -------------- | ----- | ------- | ---------------- | ----------------
       * [ 1, 0, 0, 1 ] |   1   |    2    | [ 1, 0, 1, 1 ]   | overwrite keys[2]
       * [ 1, 0, 1, 1 ] |   0   |    0    | [ 0, 0, 1, 1 ]   |
       * [ 0, 0, 1, 1 ] |   1   |    3    | [ 0, 0, 1, 1 ]   | extra key
       * [ 0, 0, 1, 1 ] |   1   |    4    | [ 0, 0, 1, 1 ] 1 | OOB access
       *
       * To avoid these issues, we have to use extra memory. The extra memory
       * holds temporary storage for writing intermediate results of each stage.
       * Since we iterate over digits in keys, we potentially need:
       * `sizeof(KeyT) * num_items * cuda::ceil_div(sizeof(KeyT),RADIX_BITS)`
       * auxiliary memory bytes. To reduce the auxiliary memory storage
       * requirements, the algorithm relies on a double buffer facility. The
       * idea behind it is in swapping destination and source buffers at each
       * iteration. This way, we can use only two buffers. One of these buffers
       * can be the final algorithm output destination. Therefore, only one
       * auxiliary array is needed. Depending on the number of iterations, we
       * can initialize the double buffer so that the algorithm output array
       * will match the double buffer result one at the final iteration.
       * A user can provide this algorithm with a double buffer straightaway to
       * further reduce the auxiliary memory requirements. `is_overwrite_okay`
       * indicates this use case.
       */
      detail::device_double_buffer<KeyT> d_keys_double_buffer(
        (is_overwrite_okay || is_num_passes_odd) ? d_keys.Alternate() : keys_allocation.get(),
        (is_overwrite_okay)   ? d_keys.Current()
        : (is_num_passes_odd) ? keys_allocation.get()
                              : d_keys.Alternate());

      detail::device_double_buffer<ValueT> d_values_double_buffer(
        (is_overwrite_okay || is_num_passes_odd) ? d_values.Alternate() : values_allocation.get(),
        (is_overwrite_okay)   ? d_values.Current()
        : (is_num_passes_odd) ? values_allocation.get()
                              : d_values.Alternate());

      using MaxPolicyT = typename PolicyHub::MaxPolicy;

      if (partition_segments)
      {
        // Partition input segments into size groups and assign specialized
        // kernels for each of them.
        error = SortWithPartitioning<LargeSegmentPolicyT, SmallAndMediumPolicyT>(
          detail::segmented_sort::DeviceSegmentedSortKernelLarge<
            Order,
            MaxPolicyT,
            KeyT,
            ValueT,
            BeginOffsetIteratorT,
            EndOffsetIteratorT,
            OffsetT>,
          detail::segmented_sort::DeviceSegmentedSortKernelSmall<
            Order,
            MaxPolicyT,
            KeyT,
            ValueT,
            BeginOffsetIteratorT,
            EndOffsetIteratorT,
            OffsetT>,
          three_way_partition_temp_storage_bytes,
          d_keys_double_buffer,
          d_values_double_buffer,
          large_segments_selector,
          small_segments_selector,
          device_partition_temp_storage,
          large_and_medium_segments_indices,
          small_segments_indices,
          group_sizes);
      }
      else
      {
        // If there are not enough segments, there's no reason to spend time
        // on extra partitioning steps.

        error = SortWithoutPartitioning<LargeSegmentPolicyT>(
          detail::segmented_sort::DeviceSegmentedSortFallbackKernel<
            Order,
            MaxPolicyT,
            KeyT,
            ValueT,
            BeginOffsetIteratorT,
            EndOffsetIteratorT,
            OffsetT>,
          d_keys_double_buffer,
          d_values_double_buffer);
      }

      d_keys.selector   = GetFinalSelector(d_keys.selector, radix_bits);
      d_values.selector = GetFinalSelector(d_values.selector, radix_bits);

    } while (false);

    return error;
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    DoubleBuffer<KeyT>& d_keys,
    DoubleBuffer<ValueT>& d_values,
    ::cuda::std::int64_t num_items,
    global_segment_offset_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    bool is_overwrite_okay,
    cudaStream_t stream)
  {
    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Create dispatch functor
      DispatchSegmentedSort dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_keys,
        d_values,
        num_items,
        num_segments,
        d_begin_offsets,
        d_end_offsets,
        is_overwrite_okay,
        stream);

      // Dispatch to chained policy
      error = CubDebug(PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (false);

    return error;
  }

private:
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE int GetNumPasses(int radix_bits)
  {
    constexpr int byte_size = 8;
    constexpr int num_bits  = sizeof(KeyT) * byte_size;
    const int num_passes    = ::cuda::ceil_div(num_bits, radix_bits);
    return num_passes;
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE int GetFinalSelector(int selector, int radix_bits)
  {
    // Sorted data always ends up in the other vector
    if (!is_overwrite_okay)
    {
      return (selector + 1) & 1;
    }

    return (selector + GetNumPasses(radix_bits)) & 1;
  }

  template <typename T>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE T* GetFinalOutput(int radix_bits, DoubleBuffer<T>& buffer)
  {
    const int final_selector = GetFinalSelector(buffer.selector, radix_bits);
    return buffer.d_buffers[final_selector];
  }

  template <typename LargeSegmentPolicyT, typename SmallAndMediumPolicyT, typename LargeKernelT, typename SmallKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t SortWithPartitioning(
    LargeKernelT large_kernel,
    SmallKernelT small_kernel,
    size_t three_way_partition_temp_storage_bytes,
    cub::detail::device_double_buffer<KeyT>& d_keys_double_buffer,
    cub::detail::device_double_buffer<ValueT>& d_values_double_buffer,
    LargeSegmentsSelectorT& large_segments_selector,
    SmallSegmentsSelectorT& small_segments_selector,
    cub::detail::temporary_storage::alias<uint8_t>& device_partition_temp_storage,
    cub::detail::temporary_storage::alias<local_segment_index_t>& large_and_medium_segments_indices,
    cub::detail::temporary_storage::alias<local_segment_index_t>& small_segments_indices,
    cub::detail::temporary_storage::alias<local_segment_index_t>& group_sizes)
  {
    cudaError_t error = cudaSuccess;

    constexpr global_segment_offset_t num_segments_per_invocation_limit =
      static_cast<global_segment_offset_t>(::cuda::std::numeric_limits<int>::max());

    // We repeatedly invoke the partitioning and sorting kernels until all segments are processed.
    const global_segment_offset_t num_invocations =
      ::cuda::ceil_div(static_cast<global_segment_offset_t>(num_segments), num_segments_per_invocation_limit);
    for (global_segment_offset_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
    {
      const global_segment_offset_t current_seg_offset = invocation_index * num_segments_per_invocation_limit;
      const local_segment_index_t current_num_segments =
        (invocation_index == (num_invocations - 1))
          ? static_cast<local_segment_index_t>(num_segments - current_seg_offset)
          : num_segments_per_invocation_limit;

      large_segments_selector.base_segment_offset = current_seg_offset;
      small_segments_selector.base_segment_offset = current_seg_offset;
      [[maybe_unused]] auto current_begin_offset  = d_begin_offsets + current_seg_offset;
      [[maybe_unused]] auto current_end_offset    = d_end_offsets + current_seg_offset;

      auto medium_indices_iterator =
        THRUST_NS_QUALIFIER::make_reverse_iterator(large_and_medium_segments_indices.get() + current_num_segments);

      error = CubDebug(cub::DevicePartition::IfNoNVTX(
        device_partition_temp_storage.get(),
        three_way_partition_temp_storage_bytes,
        THRUST_NS_QUALIFIER::counting_iterator<local_segment_index_t>(0),
        large_and_medium_segments_indices.get(),
        small_segments_indices.get(),
        medium_indices_iterator,
        group_sizes.get(),
        current_num_segments,
        large_segments_selector,
        small_segments_selector,
        stream));
      if (cudaSuccess != error)
      {
        return error;
      }

      // The device path is only used (and only compiles) when CDP is enabled.
      // It's defined in a macro since we can't put `#ifdef`s inside of
      // `NV_IF_TARGET`.
#ifndef CUB_RDC_ENABLED

#  define CUB_TEMP_DEVICE_CODE

#else // CUB_RDC_ENABLED

#  define CUB_TEMP_DEVICE_CODE                                               \
    error =                                                                  \
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(1, 1, 0, stream) \
        .doit(                                                               \
          detail::segmented_sort::DeviceSegmentedSortContinuationKernel<     \
            typename PolicyHub::MaxPolicy,                                   \
            LargeKernelT,                                                    \
            SmallKernelT,                                                    \
            KeyT,                                                            \
            ValueT,                                                          \
            BeginOffsetIteratorT,                                            \
            EndOffsetIteratorT>,                                             \
          large_kernel,                                                      \
          small_kernel,                                                      \
          current_num_segments,                                              \
          d_keys.Current(),                                                  \
          GetFinalOutput<KeyT>(LargeSegmentPolicyT::RADIX_BITS, d_keys),     \
          d_keys_double_buffer,                                              \
          d_values.Current(),                                                \
          GetFinalOutput<ValueT>(LargeSegmentPolicyT::RADIX_BITS, d_values), \
          d_values_double_buffer,                                            \
          current_begin_offset,                                              \
          current_end_offset,                                                \
          group_sizes.get(),                                                 \
          large_and_medium_segments_indices.get(),                           \
          small_segments_indices.get());                                     \
    error = CubDebug(error);                                                 \
                                                                             \
    if (cudaSuccess != error)                                                \
    {                                                                        \
      return error;                                                          \
    }                                                                        \
                                                                             \
    error = CubDebug(detail::DebugSyncStream(stream));                       \
    if (cudaSuccess != error)                                                \
    {                                                                        \
      return error;                                                          \
    }

#endif // CUB_RDC_ENABLED

      // Clang format mangles some of this NV_IF_TARGET block
      // clang-format off
    NV_IF_TARGET(
      NV_IS_HOST,
      (
        local_segment_index_t h_group_sizes[num_selected_groups];
        error = CubDebug(cudaMemcpyAsync(h_group_sizes,
                                             group_sizes.get(),
                                             num_selected_groups *
                                               sizeof(local_segment_index_t),
                                             cudaMemcpyDeviceToHost,
                                             stream));

        if (cudaSuccess != error)
        {
            return error;
        }

        error = CubDebug(SyncStream(stream));
        if (cudaSuccess != error)
        {
          return error;
        }

        error = detail::segmented_sort::DeviceSegmentedSortContinuation<LargeSegmentPolicyT,
                                                SmallAndMediumPolicyT>(
          large_kernel,
          small_kernel,
          current_num_segments,
          d_keys.Current(),
          GetFinalOutput<KeyT>(LargeSegmentPolicyT::RADIX_BITS, d_keys),
          d_keys_double_buffer,
          d_values.Current(),
          GetFinalOutput<ValueT>(LargeSegmentPolicyT::RADIX_BITS, d_values),
          d_values_double_buffer,
          current_begin_offset,
          current_end_offset,
          h_group_sizes,
          large_and_medium_segments_indices.get(),
          small_segments_indices.get(),
          stream);),
      // NV_IS_DEVICE:
      (CUB_TEMP_DEVICE_CODE));
      // clang-format on
    }
#undef CUB_TEMP_DEVICE_CODE

    return error;
  }

  template <typename LargeSegmentPolicyT, typename FallbackKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t SortWithoutPartitioning(
    FallbackKernelT fallback_kernel,
    cub::detail::device_double_buffer<KeyT>& d_keys_double_buffer,
    cub::detail::device_double_buffer<ValueT>& d_values_double_buffer)
  {
    cudaError_t error = cudaSuccess;

    const auto blocks_in_grid       = static_cast<local_segment_index_t>(num_segments);
    constexpr auto threads_in_block = static_cast<unsigned int>(LargeSegmentPolicyT::BLOCK_THREADS);

// Log kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking DeviceSegmentedSortFallbackKernel<<<%d, %d, "
            "0, %lld>>>(), %d items per thread, bit_grain %d\n",
            blocks_in_grid,
            threads_in_block,
            (long long) stream,
            LargeSegmentPolicyT::ITEMS_PER_THREAD,
            LargeSegmentPolicyT::RADIX_BITS);
#endif // CUB_DEBUG_LOG

    // Invoke fallback kernel
    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(blocks_in_grid, threads_in_block, 0, stream)
      .doit(fallback_kernel,
            d_keys.Current(),
            GetFinalOutput(LargeSegmentPolicyT::RADIX_BITS, d_keys),
            d_keys_double_buffer,
            d_values.Current(),
            GetFinalOutput(LargeSegmentPolicyT::RADIX_BITS, d_values),
            d_values_double_buffer,
            d_begin_offsets,
            d_end_offsets);

    // Check for failure to launch
    error = CubDebug(cudaPeekAtLastError());
    if (cudaSuccess != error)
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    error = CubDebug(detail::DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      return error;
    }

    return error;
  }
};

CUB_NAMESPACE_END
