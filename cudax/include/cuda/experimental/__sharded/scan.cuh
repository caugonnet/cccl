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
 * @brief In-place scans over sharded arrays: each place runs the device-scope
 *        primitive (CUB `DeviceScan`) on its shard, then per-place totals are
 *        prefix-combined and folded back into the shards in place over the
 *        shared address space.
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

#include <cub/device/device_scan.cuh>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/// @brief Scan flavor.
enum class scan_type
{
  inclusive, //!< output[i] = op(input[0], ..., input[i])
  exclusive //!< output[i] = op(init, input[0], ..., input[i-1])
};

namespace detail
{
template <typename _Tp, typename _ScanOp>
struct apply_prefix_fn
{
  _Tp prefix;
  _ScanOp op;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE _Tp operator()(_Tp val) const
  {
    return op(prefix, val);
  }
};

template <typename _Tp, typename _ScanOp>
void scan_impl(place_group&, sharded_array<_Tp>& data, scan_type type, _ScanOp scan_op, _Tp init_value)
{
  if (data.empty())
  {
    return;
  }

  const size_t num_shards = data.num_shards();

  // Pinned host staging for shard totals / prefixes (initialized so skipped
  // empty shards contribute the identity)
  places::place_memory_resource host_mr(data_place::host());
  const size_t host_bytes = 3 * num_shards * sizeof(_Tp);
  _Tp* h_shard_totals     = static_cast<_Tp*>(host_mr.allocate_sync(host_bytes, alignof(_Tp)));
  _Tp* h_prefixes         = h_shard_totals + num_shards;
  _Tp* h_last_elements    = h_prefixes + num_shards; // for exclusive scans
  ::std::fill(h_shard_totals, h_shard_totals + num_shards, init_value);
  ::std::fill(h_last_elements, h_last_elements + num_shards, init_value);

  // Per-shard CUB temp storage, freed after the final sync
  ::std::vector<::std::tuple<places::place_memory_resource, void*, size_t>> temp_storage;
  temp_storage.reserve(num_shards);

  // ==========================================================================
  // Phase 1: local scan on each shard
  // ==========================================================================

  data.each_shard->*[&](const size_t g, auto& s) {
    // For exclusive scans, save the last element BEFORE it is overwritten:
    // the shard total is op(exclusive_last, original_last)
    if (type == scan_type::exclusive)
    {
      cuda_safe_call(
        cudaMemcpyAsync(&h_last_elements[g], s.data + s.size - 1, sizeof(_Tp), cudaMemcpyDeviceToHost, s.stream));
    }

    // Query CUB temp storage requirements
    size_t bytes = 0;
    if (type == scan_type::inclusive)
    {
      cuda_safe_call(cub::DeviceScan::InclusiveScan(nullptr, bytes, s.data, s.data, scan_op, s.size, s.stream));
    }
    else
    {
      cuda_safe_call(
        cub::DeviceScan::ExclusiveScan(nullptr, bytes, s.data, s.data, scan_op, init_value, s.size, s.stream));
    }

    places::place_memory_resource mr(s.place);
    void* d_temp = mr.allocate(::cuda::stream_ref{s.stream}, bytes);
    temp_storage.emplace_back(mr, d_temp, bytes);

    // Run the local scan in place
    if (type == scan_type::inclusive)
    {
      cuda_safe_call(cub::DeviceScan::InclusiveScan(d_temp, bytes, s.data, s.data, scan_op, s.size, s.stream));
    }
    else
    {
      cuda_safe_call(
        cub::DeviceScan::ExclusiveScan(d_temp, bytes, s.data, s.data, scan_op, init_value, s.size, s.stream));
    }

    // The last scanned element: for inclusive scans this IS the shard total;
    // for exclusive scans the total also needs the saved original last element
    cuda_safe_call(
      cudaMemcpyAsync(&h_shard_totals[g], s.data + s.size - 1, sizeof(_Tp), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  if (type == scan_type::exclusive)
  {
    for (size_t g = 0; g < num_shards; g++)
    {
      h_shard_totals[g] = scan_op(h_shard_totals[g], h_last_elements[g]);
    }
  }

  // ==========================================================================
  // Phase 2: prefix-combine the shard totals on the host (P values)
  // ==========================================================================

  h_prefixes[0] = init_value;
  for (size_t g = 1; g < num_shards; g++)
  {
    h_prefixes[g] = scan_op(h_prefixes[g - 1], h_shard_totals[g - 1]);
  }

  // ==========================================================================
  // Phase 3: fold each shard's prefix into the shard, in place
  // ==========================================================================

  data.each_shard->*[&](const size_t g, auto& s) {
    if (g == 0)
    {
      return;
    }
    const _Tp prefix = h_prefixes[g];
    if (prefix == init_value)
    {
      return; // identity prefix: nothing to fold
    }
    thrust::transform(
      thrust::cuda::par_nosync.on(s.stream),
      s.data,
      s.data + s.size,
      s.data,
      apply_prefix_fn<_Tp, _ScanOp>{prefix, scan_op});
    cuda_safe_call(cudaGetLastError());
  };

  data.sync();

  for (auto& [mr, ptr, bytes] : temp_storage)
  {
    mr.deallocate_sync(ptr, bytes);
  }
  host_mr.deallocate_sync(h_shard_totals, host_bytes, alignof(_Tp));
}
} // namespace detail

/// @brief In-place inclusive scan with a custom operator.
template <typename _Tp, typename _ScanOp>
void inclusive_scan(place_group& group, sharded_array<_Tp>& data, _ScanOp scan_op)
{
  detail::scan_impl<_Tp>(group, data, scan_type::inclusive, scan_op, _Tp{0});
}

/// @brief In-place inclusive sum.
template <typename _Tp>
void inclusive_scan(place_group& group, sharded_array<_Tp>& data)
{
  detail::scan_impl<_Tp>(group, data, scan_type::inclusive, ::cuda::std::plus<_Tp>{}, _Tp{0});
}

/// @brief In-place exclusive scan with a custom operator and initial value.
template <typename _Tp,
          typename _ScanOp,
          typename = ::cuda::std::enable_if_t<::cuda::std::is_invocable_v<_ScanOp, _Tp, _Tp>>>
void exclusive_scan(place_group& group, sharded_array<_Tp>& data, _ScanOp scan_op, _Tp init_value = _Tp{0})
{
  detail::scan_impl<_Tp>(group, data, scan_type::exclusive, scan_op, init_value);
}

/// @brief In-place exclusive sum.
template <typename _Tp>
void exclusive_scan(place_group& group, sharded_array<_Tp>& data, _Tp init_value = _Tp{0})
{
  detail::scan_impl<_Tp>(group, data, scan_type::exclusive, ::cuda::std::plus<_Tp>{}, init_value);
}

/// @brief Alias for the inclusive sum.
template <typename _Tp>
void inclusive_sum(place_group& group, sharded_array<_Tp>& data)
{
  inclusive_scan(group, data);
}

/// @brief Alias for the exclusive sum.
template <typename _Tp>
void exclusive_sum(place_group& group, sharded_array<_Tp>& data, _Tp init_value = _Tp{0})
{
  exclusive_scan(group, data, init_value);
}
} // namespace cuda::experimental::sharded
