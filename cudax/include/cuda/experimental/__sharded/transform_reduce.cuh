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
 * @brief Fused transform-then-reduce over sharded arrays: each place runs
 *        `cub::DeviceReduce::TransformReduce` on its shard (one kernel, no
 *        materialized intermediate), then the per-place partials are
 *        combined — same structure as `sharded::reduce`/`reduce_into`.
 *
 * `zip_transform_reduce(_into)` is the two-input form: the per-shard input
 * handed to CUB is a `zip_iterator` over the two shards' pointers, so a
 * binary combine (e.g. `b[i] - Ax[i]`) fuses into the reduce without ever
 * writing the combined array.
 *
 * Every entry point here takes `identity` (CUB's own per-shard seed — a true
 * algebraic identity of `reduce_op`, applied once PER SHARD, harmless to
 * duplicate because a real identity has no effect) separately from an
 * optional `init` (the sharded array's overall starting value, applied
 * exactly ONCE, in the combine step — same shape as `reduce_into`; see
 * `reduce.cuh` for the full rationale).
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

#include <cuda/__iterator/zip_iterator.h>
#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/tuple>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__sharded/reduce.cuh> // __detail::__reduce_into_combine(_with_init)
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/**
 * @brief Fused unary transform + reduce: result = fold(reduce_op, transform_op(data[i])).
 *
 * Phase 1 runs `cub::DeviceReduce::TransformReduce` per shard (one kernel,
 * `transform_op` never materialized); phase 2 combines the per-place
 * partials, seeded from the FIRST partial (no externally supplied value —
 * see the `init` overload below for the exactly-once starting-value shape).
 * SYNCHRONOUS: returns the final value.
 *
 * @param group        the place group providing per-place memory resources
 * @param data         the sharded input (not modified)
 * @param transform_op host- and device-callable unary functor
 * @param reduce_op    host- and device-callable binary operator
 * @param identity     a true algebraic identity of @p reduce_op in the
 *                     RESULT type (`reduce_op(identity, x) == x`); required
 *                     by CUB's per-shard call, applied once PER SHARD.
 */
template <typename _Tp, typename _TransformOp, typename _ReduceOp, typename _Up>
[[nodiscard]] _CCCL_HOST_API _Up transform_reduce(
  place_group& group, const sharded_array<_Tp>& data, _TransformOp transform_op, _ReduceOp reduce_op, _Up identity = _Up{})
{
  if (data.empty())
  {
    return identity;
  }

  const size_t num_shards = data.num_shards();

  places::place_memory_resource host_mr(data_place::host());
  _Up* h_partials = static_cast<_Up*>(host_mr.allocate_sync(num_shards * sizeof(_Up), alignof(_Up)));

  ::std::vector<::std::pair<places::place_memory_resource, _Up*>> d_outputs;
  d_outputs.reserve(num_shards);

  data.each_shard->*[&](const size_t g, const auto& s) {
    places::place_memory_resource mr(s.place);
    _Up* d_out = static_cast<_Up*>(mr.allocate(::cuda::stream_ref{s.stream}, sizeof(_Up), alignof(_Up)));
    d_outputs.emplace_back(mr, d_out);

    const auto env = group.env(s.place, s.stream);
    cuda_safe_call(cub::DeviceReduce::TransformReduce(s.data, d_out, s.size, reduce_op, transform_op, identity, env));

    cuda_safe_call(cudaMemcpyAsync(&h_partials[g], d_out, sizeof(_Up), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  _Up result = h_partials[0];
  for (size_t g = 1; g < num_shards; g++)
  {
    result = reduce_op(result, h_partials[g]);
  }

  for (auto& [mr, ptr] : d_outputs)
  {
    mr.deallocate_sync(ptr, sizeof(_Up), alignof(_Up));
  }
  host_mr.deallocate_sync(h_partials, num_shards * sizeof(_Up), alignof(_Up));

  return result;
}

/**
 * @brief Fused unary transform + reduce, starting from @p init:
 *        `result = fold(reduce_op, init, transform_op(data[0..n)))`.
 *
 * Same as the four-argument `transform_reduce`, except @p init is applied
 * exactly ONCE — after the per-shard partials — so, unlike @p identity, it
 * need not be an algebraic identity of @p reduce_op.
 */
template <typename _Tp, typename _TransformOp, typename _ReduceOp, typename _Up>
[[nodiscard]] _CCCL_HOST_API _Up transform_reduce(
  place_group& group,
  const sharded_array<_Tp>& data,
  _TransformOp transform_op,
  _ReduceOp reduce_op,
  _Up identity,
  _Up init)
{
  if (data.empty())
  {
    return init;
  }

  const size_t num_shards = data.num_shards();

  places::place_memory_resource host_mr(data_place::host());
  _Up* h_partials = static_cast<_Up*>(host_mr.allocate_sync(num_shards * sizeof(_Up), alignof(_Up)));

  ::std::vector<::std::pair<places::place_memory_resource, _Up*>> d_outputs;
  d_outputs.reserve(num_shards);

  data.each_shard->*[&](const size_t g, const auto& s) {
    places::place_memory_resource mr(s.place);
    _Up* d_out = static_cast<_Up*>(mr.allocate(::cuda::stream_ref{s.stream}, sizeof(_Up), alignof(_Up)));
    d_outputs.emplace_back(mr, d_out);

    const auto env = group.env(s.place, s.stream);
    cuda_safe_call(cub::DeviceReduce::TransformReduce(s.data, d_out, s.size, reduce_op, transform_op, identity, env));

    cuda_safe_call(cudaMemcpyAsync(&h_partials[g], d_out, sizeof(_Up), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  _Up result = init;
  for (size_t g = 0; g < num_shards; g++)
  {
    result = reduce_op(result, h_partials[g]);
  }

  for (auto& [mr, ptr] : d_outputs)
  {
    mr.deallocate_sync(ptr, sizeof(_Up), alignof(_Up));
  }
  host_mr.deallocate_sync(h_partials, num_shards * sizeof(_Up), alignof(_Up));

  return result;
}

namespace __detail
{
// Adapts a two-argument zip_op(a, b) into the unary functor CUB's
// TransformReduce expects, called on a zip_iterator's `tuple<a&, b&>`.
template <typename _ZipOp>
struct __zip_unpack
{
  _ZipOp __op;

  template <typename _Tuple>
  _CCCL_HOST_DEVICE_API auto operator()(_Tuple&& __t) const
    -> decltype(__op(::cuda::std::get<0>(__t), ::cuda::std::get<1>(__t)))
  {
    return __op(::cuda::std::get<0>(__t), ::cuda::std::get<1>(__t));
  }
};
} // namespace __detail

/**
 * @brief Fused binary (zip) transform + reduce: result = fold(reduce_op, zip_op(input1[i], input2[i])).
 *
 * Same structure as `transform_reduce`, except the per-shard input handed to
 * CUB is a `zip_iterator` over both shards' pointers: no intermediate array
 * is ever written (contrast with `sharded::transform(binary)` followed by
 * `sharded::reduce`, which is two passes through memory). Combine is seeded
 * from the first partial, no externally supplied identity for the combine
 * itself (see the `init` overload for the exactly-once starting value).
 *
 * @throws std::invalid_argument when the two inputs' layouts are not compatible
 */
template <typename _Tp, typename _Up, typename _ZipOp, typename _ReduceOp, typename _Vp>
[[nodiscard]] _CCCL_HOST_API _Vp zip_transform_reduce(
  place_group& group,
  const sharded_array<_Tp>& input1,
  const sharded_array<_Up>& input2,
  _ZipOp zip_op,
  _ReduceOp reduce_op,
  _Vp identity = _Vp{})
{
  check_compatible(input1, input2, "zip_transform_reduce");

  if (input1.empty())
  {
    return identity;
  }

  const size_t num_shards = input1.num_shards();

  places::place_memory_resource host_mr(data_place::host());
  _Vp* h_partials = static_cast<_Vp*>(host_mr.allocate_sync(num_shards * sizeof(_Vp), alignof(_Vp)));

  ::std::vector<::std::pair<places::place_memory_resource, _Vp*>> d_outputs;
  d_outputs.reserve(num_shards);

  const __detail::__zip_unpack<_ZipOp> unpacked_op{zip_op};

  input1.each_shard->*[&](const size_t g, const auto& s1) {
    const auto& s2 = input2.shard(g);
    places::place_memory_resource mr(s1.place);
    _Vp* d_out = static_cast<_Vp*>(mr.allocate(::cuda::stream_ref{s1.stream}, sizeof(_Vp), alignof(_Vp)));
    d_outputs.emplace_back(mr, d_out);

    const auto env  = group.env(s1.place, s1.stream);
    const auto d_in = ::cuda::make_zip_iterator(s1.data, s2.data);
    cuda_safe_call(cub::DeviceReduce::TransformReduce(d_in, d_out, s1.size, reduce_op, unpacked_op, identity, env));

    cuda_safe_call(cudaMemcpyAsync(&h_partials[g], d_out, sizeof(_Vp), cudaMemcpyDeviceToHost, s1.stream));
  };

  input1.sync();

  _Vp result = h_partials[0];
  for (size_t g = 1; g < num_shards; g++)
  {
    result = reduce_op(result, h_partials[g]);
  }

  for (auto& [mr, ptr] : d_outputs)
  {
    mr.deallocate_sync(ptr, sizeof(_Vp), alignof(_Vp));
  }
  host_mr.deallocate_sync(h_partials, num_shards * sizeof(_Vp), alignof(_Vp));

  return result;
}

/**
 * @brief Fused binary (zip) transform + reduce, starting from @p init:
 *        `result = fold(reduce_op, init, zip_op(input1[0..n), input2[0..n)))`.
 *
 * Same as the five-argument `zip_transform_reduce`, except @p init is
 * applied exactly ONCE, after the per-shard partials.
 */
template <typename _Tp, typename _Up, typename _ZipOp, typename _ReduceOp, typename _Vp>
[[nodiscard]] _CCCL_HOST_API _Vp zip_transform_reduce(
  place_group& group,
  const sharded_array<_Tp>& input1,
  const sharded_array<_Up>& input2,
  _ZipOp zip_op,
  _ReduceOp reduce_op,
  _Vp identity,
  _Vp init)
{
  check_compatible(input1, input2, "zip_transform_reduce");

  if (input1.empty())
  {
    return init;
  }

  const size_t num_shards = input1.num_shards();

  places::place_memory_resource host_mr(data_place::host());
  _Vp* h_partials = static_cast<_Vp*>(host_mr.allocate_sync(num_shards * sizeof(_Vp), alignof(_Vp)));

  ::std::vector<::std::pair<places::place_memory_resource, _Vp*>> d_outputs;
  d_outputs.reserve(num_shards);

  const __detail::__zip_unpack<_ZipOp> unpacked_op{zip_op};

  input1.each_shard->*[&](const size_t g, const auto& s1) {
    const auto& s2 = input2.shard(g);
    places::place_memory_resource mr(s1.place);
    _Vp* d_out = static_cast<_Vp*>(mr.allocate(::cuda::stream_ref{s1.stream}, sizeof(_Vp), alignof(_Vp)));
    d_outputs.emplace_back(mr, d_out);

    const auto env  = group.env(s1.place, s1.stream);
    const auto d_in = ::cuda::make_zip_iterator(s1.data, s2.data);
    cuda_safe_call(cub::DeviceReduce::TransformReduce(d_in, d_out, s1.size, reduce_op, unpacked_op, identity, env));

    cuda_safe_call(cudaMemcpyAsync(&h_partials[g], d_out, sizeof(_Vp), cudaMemcpyDeviceToHost, s1.stream));
  };

  input1.sync();

  _Vp result = init;
  for (size_t g = 0; g < num_shards; g++)
  {
    result = reduce_op(result, h_partials[g]);
  }

  for (auto& [mr, ptr] : d_outputs)
  {
    mr.deallocate_sync(ptr, sizeof(_Vp), alignof(_Vp));
  }
  host_mr.deallocate_sync(h_partials, num_shards * sizeof(_Vp), alignof(_Vp));

  return result;
}

// ============================================================================
// _into variants: device-resident output on place 0, capture-safe. Same
// phase-1/join/combine shape as `reduce_into` (see reduce.cuh); the combine
// kernels are literally reused from there since folding partials is
// independent of how they were produced (plain reduce vs. transform_reduce
// vs. zip_transform_reduce).
// ============================================================================

namespace __detail
{
template <typename _Tp, typename _TransformOp, typename _ReduceOp, typename _Up>
_CCCL_HOST_API auto __transform_reduce_into_partials(
  place_group& group, const sharded_array<_Tp>& data, _TransformOp transform_op, _ReduceOp reduce_op, _Up identity)
{
  const size_t num_shards = data.num_shards();
  const auto& s0          = data.shard(0);

  places::place_memory_resource mr0(s0.place);
  _Up* partials = static_cast<_Up*>(mr0.allocate(::cuda::stream_ref{s0.stream}, num_shards * sizeof(_Up), alignof(_Up)));

  data.each_shard->*[&](const size_t g, const auto& s) {
    const auto env = group.env(s.place, s.stream);
    cuda_safe_call(cub::DeviceReduce::TransformReduce(s.data, partials + g, s.size, reduce_op, transform_op, identity, env));
  };

  data.join_into(s0.stream);
  return ::std::make_tuple(mr0, partials, num_shards, s0);
}

template <typename _Tp, typename _Up, typename _ZipOp, typename _ReduceOp, typename _Vp>
_CCCL_HOST_API auto __zip_transform_reduce_into_partials(
  place_group& group,
  const sharded_array<_Tp>& input1,
  const sharded_array<_Up>& input2,
  _ZipOp zip_op,
  _ReduceOp reduce_op,
  _Vp identity)
{
  check_compatible(input1, input2, "zip_transform_reduce_into");

  const size_t num_shards = input1.num_shards();
  const auto& s0          = input1.shard(0);

  places::place_memory_resource mr0(s0.place);
  _Vp* partials = static_cast<_Vp*>(mr0.allocate(::cuda::stream_ref{s0.stream}, num_shards * sizeof(_Vp), alignof(_Vp)));

  const __zip_unpack<_ZipOp> unpacked_op{zip_op};

  input1.each_shard->*[&](const size_t g, const auto& s1) {
    const auto& s2  = input2.shard(g);
    const auto env  = group.env(s1.place, s1.stream);
    const auto d_in = ::cuda::make_zip_iterator(s1.data, s2.data);
    cuda_safe_call(cub::DeviceReduce::TransformReduce(d_in, partials + g, s1.size, reduce_op, unpacked_op, identity, env));
  };

  input1.join_into(s0.stream);
  return ::std::make_tuple(mr0, partials, num_shards, s0);
}
} // namespace __detail

/**
 * @brief Fused unary transform + reduce, writing the result into device
 *        memory on place 0. CAPTURE-SAFE — see `reduce_into` for the full
 *        design (place-0 buffer, `join_into`, on-device combine kernel, no
 *        host round-trip). @p identity is CUB's per-shard seed only.
 */
template <typename _Tp, typename _TransformOp, typename _ReduceOp, typename _Up>
_CCCL_HOST_API void transform_reduce_into(
  place_group& group,
  const sharded_array<_Tp>& data,
  _Up* out,
  _TransformOp transform_op,
  _ReduceOp reduce_op,
  _Up identity = _Up{})
{
  if (data.empty())
  {
    cuda_safe_call(cudaMemcpy(out, &identity, sizeof(_Up), cudaMemcpyHostToDevice));
    return;
  }

  auto [mr0, partials, num_shards, s0] =
    __detail::__transform_reduce_into_partials(group, data, transform_op, reduce_op, identity);

  {
    exec_place_scope scope(s0.exec);
    __detail::__reduce_into_combine<<<1, 1, 0, s0.stream>>>(partials, num_shards, reduce_op, out);
    cuda_safe_call(cudaGetLastError());
  }

  __detail::__reduce_into_free_partials(mr0, partials, num_shards, s0);
}

/**
 * @brief Fused unary transform + reduce starting from @p init (applied
 *        exactly once), writing the result into device memory on place 0.
 */
template <typename _Tp, typename _TransformOp, typename _ReduceOp, typename _Up>
_CCCL_HOST_API void transform_reduce_into(
  place_group& group,
  const sharded_array<_Tp>& data,
  _Up* out,
  _TransformOp transform_op,
  _ReduceOp reduce_op,
  _Up identity,
  _Up init)
{
  if (data.empty())
  {
    cuda_safe_call(cudaMemcpy(out, &init, sizeof(_Up), cudaMemcpyHostToDevice));
    return;
  }

  auto [mr0, partials, num_shards, s0] =
    __detail::__transform_reduce_into_partials(group, data, transform_op, reduce_op, identity);

  {
    exec_place_scope scope(s0.exec);
    __detail::__reduce_into_combine_with_init<<<1, 1, 0, s0.stream>>>(partials, num_shards, reduce_op, init, out);
    cuda_safe_call(cudaGetLastError());
  }

  __detail::__reduce_into_free_partials(mr0, partials, num_shards, s0);
}

/**
 * @brief Fused binary (zip) transform + reduce, writing the result into
 *        device memory on place 0. CAPTURE-SAFE — same design as
 *        `transform_reduce_into`/`reduce_into`, with the two-shard
 *        zip_iterator input from `zip_transform_reduce`.
 *
 * @throws std::invalid_argument when the two inputs' layouts are not compatible
 */
template <typename _Tp, typename _Up, typename _ZipOp, typename _ReduceOp, typename _Vp>
_CCCL_HOST_API void zip_transform_reduce_into(
  place_group& group,
  const sharded_array<_Tp>& input1,
  const sharded_array<_Up>& input2,
  _Vp* out,
  _ZipOp zip_op,
  _ReduceOp reduce_op,
  _Vp identity = _Vp{})
{
  check_compatible(input1, input2, "zip_transform_reduce_into");

  if (input1.empty())
  {
    cuda_safe_call(cudaMemcpy(out, &identity, sizeof(_Vp), cudaMemcpyHostToDevice));
    return;
  }

  auto [mr0, partials, num_shards, s0] =
    __detail::__zip_transform_reduce_into_partials(group, input1, input2, zip_op, reduce_op, identity);

  {
    exec_place_scope scope(s0.exec);
    __detail::__reduce_into_combine<<<1, 1, 0, s0.stream>>>(partials, num_shards, reduce_op, out);
    cuda_safe_call(cudaGetLastError());
  }

  __detail::__reduce_into_free_partials(mr0, partials, num_shards, s0);
}

/**
 * @brief Fused binary (zip) transform + reduce starting from @p init
 *        (applied exactly once), writing the result into device memory on
 *        place 0.
 *
 * @throws std::invalid_argument when the two inputs' layouts are not compatible
 */
template <typename _Tp, typename _Up, typename _ZipOp, typename _ReduceOp, typename _Vp>
_CCCL_HOST_API void zip_transform_reduce_into(
  place_group& group,
  const sharded_array<_Tp>& input1,
  const sharded_array<_Up>& input2,
  _Vp* out,
  _ZipOp zip_op,
  _ReduceOp reduce_op,
  _Vp identity,
  _Vp init)
{
  check_compatible(input1, input2, "zip_transform_reduce_into");

  if (input1.empty())
  {
    cuda_safe_call(cudaMemcpy(out, &init, sizeof(_Vp), cudaMemcpyHostToDevice));
    return;
  }

  auto [mr0, partials, num_shards, s0] =
    __detail::__zip_transform_reduce_into_partials(group, input1, input2, zip_op, reduce_op, identity);

  {
    exec_place_scope scope(s0.exec);
    __detail::__reduce_into_combine_with_init<<<1, 1, 0, s0.stream>>>(partials, num_shards, reduce_op, init, out);
    cuda_safe_call(cudaGetLastError());
  }

  __detail::__reduce_into_free_partials(mr0, partials, num_shards, s0);
}
} // namespace cuda::experimental::sharded
