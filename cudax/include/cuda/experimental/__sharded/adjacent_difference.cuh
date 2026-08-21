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
 * @brief Adjacent difference over sharded arrays. Each shard computes its
 *        differences locally; the only cross-place traffic is one boundary
 *        element per shard (the predecessor of the shard's first element).
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

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace detail
{
/**
 * @brief Per-shard adjacent difference kernel.
 *
 * output[i] = op(input[i], input[i-1]) for i > 0.
 * output[0] = op(input[0], *prev_last) when a predecessor exists (pinned host
 * boundary element from the previous shard), otherwise input[0].
 */
template <typename _Tp, typename _BinaryOp>
__global__ void adjacent_difference_kernel(const _Tp* input, _Tp* output, size_t n, const _Tp* prev_last, _BinaryOp op)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= n)
  {
    return;
  }

  if (idx == 0)
  {
    output[0] = prev_last ? op(input[0], *prev_last) : input[0];
  }
  else
  {
    output[idx] = op(input[idx], input[idx - 1]);
  }
}
} // namespace detail

/**
 * @brief Out-of-place adjacent difference with a custom binary operator:
 * output[i] = op(input[i], input[i-1]), output[0] = input[0].
 *
 * Input and output must be compatible (same shard sizes and places) and must
 * not alias: neighboring elements are read while outputs are written.
 * SYNCHRONOUS.
 *
 * @throws std::invalid_argument when the layouts are not compatible
 */
template <typename _Tp, typename _BinaryOp>
void adjacent_difference(place_group&, sharded_array<_Tp>& input, sharded_array<_Tp>& output, _BinaryOp op)
{
  check_compatible(input, output, "adjacent_difference");

  const size_t num_shards = output.num_shards();
  if (num_shards == 0 || input.size() == 0)
  {
    return;
  }

  // Boundary-element staging requires host synchronization: cannot be captured
  detail::check_not_capturing(input, "sharded::adjacent_difference");
  detail::check_not_capturing(output, "sharded::adjacent_difference");

  // Pinned host buffer for the per-shard boundary elements: written once per
  // shard, read (zero-copy) by the successor shard's kernel
  places::place_memory_resource host_mr(data_place::host());
  _Tp* h_last_elements = static_cast<_Tp*>(host_mr.allocate_sync(num_shards * sizeof(_Tp), alignof(_Tp)));

  // Phase 1: gather each shard's last element
  input.sync();
  input.each_shard->*[h_last_elements](size_t g, auto& in_shard) {
    cuda_safe_call(cudaMemcpyAsync(
      &h_last_elements[g], in_shard.data + in_shard.size - 1, sizeof(_Tp), cudaMemcpyDeviceToHost, in_shard.stream));
  };
  input.sync();

  // Phase 2: per-shard differences (boundary from the predecessor shard)
  output.sync();
  output.each_shard->*[&input, h_last_elements, op](size_t g, auto& out_shard) {
    constexpr int block_size = 256;
    const auto& in_shard     = input.shard(g);

    const _Tp* prev_last = (g > 0) && (input.shard(g - 1).size > 0) ? &h_last_elements[g - 1] : nullptr;
    const int num_blocks = static_cast<int>((out_shard.size + block_size - 1) / block_size);

    detail::adjacent_difference_kernel<<<num_blocks, block_size, 0, out_shard.stream>>>(
      in_shard.data, out_shard.data, out_shard.size, prev_last, op);
    cuda_safe_call(cudaGetLastError());
  };

  output.sync();
  host_mr.deallocate_sync(h_last_elements, num_shards * sizeof(_Tp), alignof(_Tp));
}

/// @brief Out-of-place adjacent difference with subtraction.
template <typename _Tp>
void adjacent_difference(place_group& group, sharded_array<_Tp>& input, sharded_array<_Tp>& output)
{
  adjacent_difference(group, input, output, ::cuda::std::minus<_Tp>{});
}
} // namespace cuda::experimental::sharded
