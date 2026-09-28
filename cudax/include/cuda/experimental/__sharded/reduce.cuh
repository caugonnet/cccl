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
 * @brief Reduction over sharded arrays: each place runs the device-scope
 *        primitive (CUB `DeviceReduce`) on its shard, then the per-place
 *        partials are combined — the same local-primitive-plus-combine
 *        structure the device scope itself uses over blocks.
 *
 * Algorithm temporaries are drawn from each shard's own place through the
 * group's per-place memory resources, so scratch lands where the work runs.
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

#include <cub/device/device_reduce.cuh>

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/limits>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/**
 * @brief Reduce all elements with a custom operator.
 *
 * Phase 1 runs CUB `DeviceReduce` per shard on the shard's stream, with
 * temporaries allocated from the shard's place; phase 2 combines the
 * per-place partials. SYNCHRONOUS: returns the final value.
 *
 * @param group   the place group providing per-place memory resources
 * @param data    the sharded input (not modified)
 * @param reduce_op host- and device-callable binary operator
 * @param init_value initial (identity) value
 */
template <typename _Tp, typename _ReduceOp>
[[nodiscard]] _CCCL_HOST_API _Tp
reduce(place_group& group, const sharded_array<_Tp>& data, _ReduceOp reduce_op, _Tp init_value = _Tp{})
{
  if (data.empty())
  {
    return init_value;
  }

  const size_t num_shards = data.num_shards();

  // Pinned host memory for the per-place partials (initialized so skipped
  // empty shards contribute the identity)
  places::place_memory_resource host_mr(data_place::host());
  _Tp* h_partials = static_cast<_Tp*>(host_mr.allocate_sync(num_shards * sizeof(_Tp), alignof(_Tp)));
  ::std::fill(h_partials, h_partials + num_shards, init_value);

  // Phase 1: local reduce on each shard; free the per-shard outputs only
  // after the final sync (places without stream-ordered deallocation)
  ::std::vector<::std::pair<places::place_memory_resource, _Tp*>> d_outputs;
  d_outputs.reserve(num_shards);

  data.each_shard->*[&](const size_t g, const auto& s) {
    places::place_memory_resource mr(s.place);
    _Tp* d_out = static_cast<_Tp*>(mr.allocate(::cuda::stream_ref{s.stream}, sizeof(_Tp), alignof(_Tp)));
    d_outputs.emplace_back(mr, d_out);

    // Temporaries come from the shard's place through the group's resources
    const auto env = group.env(s.place, s.stream);
    cuda_safe_call(cub::DeviceReduce::Reduce(s.data, d_out, s.size, reduce_op, init_value, env));

    cuda_safe_call(cudaMemcpyAsync(&h_partials[g], d_out, sizeof(_Tp), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  // Phase 2: combine the per-place partials
  _Tp result = init_value;
  for (size_t g = 0; g < num_shards; g++)
  {
    result = reduce_op(result, h_partials[g]);
  }

  for (auto& [mr, ptr] : d_outputs)
  {
    mr.deallocate_sync(ptr, sizeof(_Tp), alignof(_Tp));
  }
  host_mr.deallocate_sync(h_partials, num_shards * sizeof(_Tp), alignof(_Tp));

  return result;
}

/// @brief Sum of all elements.
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp sum(place_group& group, const sharded_array<_Tp>& data)
{
  return reduce(group, data, ::cuda::std::plus<_Tp>{}, _Tp{0});
}

/// @brief Minimum element.
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp min(place_group& group, const sharded_array<_Tp>& data)
{
  return reduce(group, data, ::cuda::minimum<_Tp>{}, ::cuda::std::numeric_limits<_Tp>::max());
}

/// @brief Maximum element.
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp max(place_group& group, const sharded_array<_Tp>& data)
{
  return reduce(group, data, ::cuda::maximum<_Tp>{}, ::cuda::std::numeric_limits<_Tp>::lowest());
}

namespace __detail
{
// Folds the per-shard partials (already resident on place 0) into *__out,
// seeded from __partials[0] itself: no identity needed here, because with
// num_shards >= 1 there is always at least one real value to start from.
template <typename _Tp, typename _ReduceOp>
__global__ void __reduce_into_combine(const _Tp* __partials, size_t __num_shards, _ReduceOp __reduce_op, _Tp* __out)
{
  _Tp __result = __partials[0];
  for (size_t __i = 1; __i < __num_shards; __i++)
  {
    __result = __reduce_op(__result, __partials[__i]);
  }
  *__out = __result;
}

// Same fold, but seeded from a caller-supplied __init instead of
// __partials[0]: __init is combined exactly ONCE here, not per shard, so it
// need not be an algebraic identity of __reduce_op (unlike __identity below).
template <typename _Tp, typename _ReduceOp>
__global__ void
__reduce_into_combine_with_init(
  const _Tp* __partials, size_t __num_shards, _ReduceOp __reduce_op, _Tp __init, _Tp* __out)
{
  _Tp __result = __init;
  for (size_t __i = 0; __i < __num_shards; __i++)
  {
    __result = __reduce_op(__result, __partials[__i]);
  }
  *__out = __result;
}

// Phase 1, shared by both reduce_into overloads: one CUB DeviceReduce per
// shard, writing straight into its slot of a buffer allocated on place 0
// (the caller frees it after consuming it). __identity seeds CUB's OWN
// per-shard reduction (required by its public API even for a non-empty
// shard) — it is applied once PER SHARD, so it must be a true algebraic
// identity of __reduce_op (reduce_op(identity, x) == x); duplicating a true
// identity across shards is a no-op, which is what makes this safe.
template <typename _Tp, typename _ReduceOp>
_CCCL_HOST_API auto __reduce_into_partials(
  place_group& group, const sharded_array<_Tp>& data, _ReduceOp reduce_op, _Tp identity)
{
  const size_t num_shards = data.num_shards();
  const auto& s0          = data.shard(0);

  places::place_memory_resource mr0(s0.place);
  _Tp* partials = static_cast<_Tp*>(mr0.allocate(::cuda::stream_ref{s0.stream}, num_shards * sizeof(_Tp), alignof(_Tp)));

  data.each_shard->*[&](const size_t g, const auto& s) {
    const auto env = group.env(s.place, s.stream);
    cuda_safe_call(cub::DeviceReduce::Reduce(s.data, partials + g, s.size, reduce_op, identity, env));
  };

  data.join_into(s0.stream);
  return ::std::make_tuple(mr0, partials, num_shards, s0);
}

template <typename _Tp, typename _Mr, typename _Shard>
_CCCL_HOST_API void __reduce_into_free_partials(_Mr& mr0, _Tp* partials, size_t num_shards, const _Shard& s0)
{
  mr0.deallocate(::cuda::stream_ref{s0.stream}, partials, num_shards * sizeof(_Tp), alignof(_Tp));
}
} // namespace __detail

/**
 * @brief Reduce all elements, writing the result into device memory on place 0.
 *
 * CAPTURE-SAFE, unlike `reduce()`: nothing is synchronized and no value is
 * read back to the host. Each shard's CUB `DeviceReduce::Reduce` writes its
 * partial directly into a slot of a single buffer allocated on place 0 (the
 * FIRST shard's place) — a remote store for every shard but the first, cheap
 * because it moves one element, not bulk data. `data.join_into` orders place
 * 0's stream after every shard stream, then one small on-device kernel folds
 * the partials and writes the result to @p out. The scratch buffer is freed
 * with a stream-ordered (not synchronous) deallocation once the combine
 * kernel has consumed it.
 *
 * @param group    the place group providing per-place memory resources
 * @param data     the sharded input (not modified)
 * @param out      device pointer, resident on place 0 (`data.shard(0).place`)
 * @param reduce_op host- and device-callable binary operator
 * @param identity a true algebraic identity of @p reduce_op
 *                 (`reduce_op(identity, x) == x` for all `x`). Required by
 *                 `cub::DeviceReduce::Reduce`'s public API for each shard's
 *                 own local reduction — applied once PER SHARD, so it must
 *                 be a real identity (`sum`/`min`/`max` already only ever
 *                 pass one); passing a non-identity value here silently
 *                 folds it in multiple times. It does NOT seed the
 *                 cross-shard combine (see the two-argument overload for
 *                 that) and is not written to @p out unless @p data is
 *                 entirely empty.
 */
template <typename _Tp, typename _ReduceOp>
_CCCL_HOST_API void
reduce_into(place_group& group, const sharded_array<_Tp>& data, _Tp* out, _ReduceOp reduce_op, _Tp identity = _Tp{})
{
  if (data.empty())
  {
    // Nothing to source a value from: this is the one case where an
    // externally supplied identity is unavoidable. No shard stream to hang
    // an async copy off of, so this one is synchronous.
    cuda_safe_call(cudaMemcpy(out, &identity, sizeof(_Tp), cudaMemcpyHostToDevice));
    return;
  }

  auto [mr0, partials, num_shards, s0] = __detail::__reduce_into_partials(group, data, reduce_op, identity);

  {
    exec_place_scope scope(s0.exec);
    __detail::__reduce_into_combine<<<1, 1, 0, s0.stream>>>(partials, num_shards, reduce_op, out);
    cuda_safe_call(cudaGetLastError());
  }

  __detail::__reduce_into_free_partials(mr0, partials, num_shards, s0);
}

/**
 * @brief Reduce all elements starting from @p init, writing the result into
 *        device memory on place 0: `result = fold(reduce_op, init, data[0..n))`.
 *
 * Same structure as the three-argument `reduce_into`, but @p init is applied
 * exactly ONCE — in the combine kernel, after the per-shard partials — so,
 * unlike @p identity, it does NOT need to be an algebraic identity of
 * @p reduce_op. This is the `std::reduce(first, last, init, op)` shape.
 *
 * @param identity a true algebraic identity of @p reduce_op, needed only for
 *                 CUB's per-shard call (see the three-argument overload).
 * @param init     the sharded array's starting value, folded in exactly once.
 */
template <typename _Tp, typename _ReduceOp>
_CCCL_HOST_API void reduce_into(
  place_group& group, const sharded_array<_Tp>& data, _Tp* out, _ReduceOp reduce_op, _Tp identity, _Tp init)
{
  if (data.empty())
  {
    cuda_safe_call(cudaMemcpy(out, &init, sizeof(_Tp), cudaMemcpyHostToDevice));
    return;
  }

  auto [mr0, partials, num_shards, s0] = __detail::__reduce_into_partials(group, data, reduce_op, identity);

  {
    exec_place_scope scope(s0.exec);
    __detail::__reduce_into_combine_with_init<<<1, 1, 0, s0.stream>>>(partials, num_shards, reduce_op, init, out);
    cuda_safe_call(cudaGetLastError());
  }

  __detail::__reduce_into_free_partials(mr0, partials, num_shards, s0);
}
} // namespace cuda::experimental::sharded
