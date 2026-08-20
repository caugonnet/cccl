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
 * @brief Shim implementations: containers and every tier-1 algorithm except
 *        sort (sort instantiates the distributed engine and lives in its own
 *        translation unit so the two compile in parallel).
 */

#include <algorithm>
#include <limits>
#include <memory>

#include "sharded_shim_impl.h"

namespace cuda_sharded_shim
{
// ==== place_group ===========================================================

pg_handle* pg_by_locality_domains(const ::std::vector<int>& device_ids)
{
  return new pg_handle{place_group::by_locality_domains(device_ids)};
}

pg_handle* pg_by_devices(const ::std::vector<int>& device_ids)
{
  return new pg_handle{place_group::by_devices(device_ids)};
}

void pg_destroy(pg_handle* pg) noexcept
{
  delete pg;
}

::std::size_t pg_size(const pg_handle* pg)
{
  return pg->group.size();
}

::std::uintptr_t pg_get_stream(pg_handle* pg, ::std::size_t place_idx, ::std::size_t color)
{
  if (place_idx >= pg->group.size())
  {
    throw ::std::invalid_argument("place_group.get_stream: place index " + ::std::to_string(place_idx)
                                  + " out of range (group has " + ::std::to_string(pg->group.size()) + " places)");
  }
  return reinterpret_cast<::std::uintptr_t>(pg->group.get_stream(place_idx, color));
}

void pg_sync(pg_handle* pg)
{
  pg->group.sync();
}

// ==== sharded_array =========================================================

namespace
{
sa_handle* make_handle(int dtype)
{
  auto* sa  = new sa_handle;
  sa->dtype = dtype;
  return sa;
}

template <typename Alloc>
sa_handle* allocate_dispatch(int dtype, Alloc&& alloc)
{
  ::std::unique_ptr<sa_handle> sa(make_handle(dtype));
  dispatch(sa.get(), [&](auto& arr) {
    using array_t = ::std::remove_reference_t<decltype(arr)>;
    arr           = alloc.template operator()<typename array_t::value_type>();
  });
  return sa.release();
}
} // namespace

sa_handle* sa_allocate(pg_handle* pg, int dtype, ::std::size_t total_size)
{
  return allocate_dispatch(dtype, [&]<typename T>() {
    return shd::sharded_array<T>::allocate(pg->group, total_size);
  });
}

sa_handle* sa_allocate_sizes(pg_handle* pg, int dtype, const ::std::vector<::std::size_t>& sizes)
{
  return allocate_dispatch(dtype, [&]<typename T>() {
    return shd::sharded_array<T>::allocate(pg->group, sizes);
  });
}

sa_handle* sa_allocate_contiguous(pg_handle* pg, int dtype, ::std::size_t total_size)
{
  return allocate_dispatch(dtype, [&]<typename T>() {
    return shd::sharded_array<T>::allocate_contiguous(pg->group, total_size);
  });
}

sa_handle* sa_adopt(pg_handle* pg,
                    int dtype,
                    const ::std::vector<::std::uintptr_t>& ptrs,
                    const ::std::vector<::std::size_t>& sizes,
                    const ::std::vector<::std::uintptr_t>& producer_streams)
{
  auto& group = pg->group;
  if (ptrs.size() != sizes.size() || ptrs.size() != producer_streams.size())
  {
    throw ::std::invalid_argument("sharded_array.adopt: ptrs, sizes and streams must have the same length");
  }
  if (ptrs.size() != group.size())
  {
    throw ::std::invalid_argument(
      "sharded_array.adopt: buffer count (" + ::std::to_string(ptrs.size())
      + ") must equal the number of places in the group (" + ::std::to_string(group.size()) + ")");
  }

  // Consumer-waits: make sure producer work on the adopted buffers is
  // complete before any group stream touches them.
  for (::std::uintptr_t s : producer_streams)
  {
    if (s != 0)
    {
      ::cuda::experimental::places::cuda_safe_call(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(s)));
    }
  }

  return allocate_dispatch(dtype, [&]<typename T>() {
    const ::std::size_t color = group.next_stream_color();
    ::std::vector<shd::shard<T>> shards;
    shards.reserve(ptrs.size());
    ::std::size_t offset = 0;
    for (::std::size_t i = 0; i < ptrs.size(); i++)
    {
      if (sizes[i] == 0)
      {
        continue;
      }
      shd::shard<T> s;
      s.data          = reinterpret_cast<T*>(ptrs[i]);
      s.size          = sizes[i];
      s.capacity      = sizes[i];
      s.global_offset = offset;
      s.place         = group.place(i).affine_data_place();
      s.exec          = group.place(i);
      s.stream        = group.get_stream(i, color);
      shards.push_back(s);
      offset += sizes[i];
    }
    return shd::sharded_array<T>(::std::move(shards)); // non-owning view
  });
}

void sa_destroy(sa_handle* sa) noexcept
{
  delete sa;
}

int sa_dtype(const sa_handle* sa)
{
  return sa->dtype;
}

::std::size_t sa_size(const sa_handle* sa)
{
  return dispatch(sa, [](const auto& arr) {
    return arr.size();
  });
}

::std::size_t sa_num_shards(const sa_handle* sa)
{
  return dispatch(sa, [](const auto& arr) {
    return arr.num_shards();
  });
}

bool sa_is_contiguous(const sa_handle* sa)
{
  return dispatch(sa, [](const auto& arr) {
    return arr.is_contiguous();
  });
}

::std::uintptr_t sa_contiguous_ptr(const sa_handle* sa)
{
  return dispatch(sa, [](const auto& arr) {
    return reinterpret_cast<::std::uintptr_t>(arr.contiguous_data());
  });
}

void sa_shard_info(
  const sa_handle* sa,
  ::std::size_t idx,
  ::std::uintptr_t* data,
  ::std::size_t* size,
  ::std::size_t* global_offset,
  ::std::uintptr_t* stream)
{
  dispatch(sa, [&](const auto& arr) {
    if (idx >= arr.num_shards())
    {
      throw ::std::out_of_range("sharded_array.shard: index " + ::std::to_string(idx) + " out of range (array has "
                                + ::std::to_string(arr.num_shards()) + " shards)");
    }
    const auto& s  = arr.shard(idx);
    *data          = reinterpret_cast<::std::uintptr_t>(s.data);
    *size          = s.size;
    *global_offset = s.global_offset;
    *stream        = reinterpret_cast<::std::uintptr_t>(s.stream);
  });
}

void sa_copy_from_host(sa_handle* sa, ::std::uintptr_t host_ptr)
{
  dispatch(sa, [&](auto& arr) {
    using T = typename ::std::remove_reference_t<decltype(arr)>::value_type;
    arr.copy_from_host(reinterpret_cast<const T*>(host_ptr));
  });
}

void sa_copy_to_host(const sa_handle* sa, ::std::uintptr_t host_ptr)
{
  dispatch(sa, [&](const auto& arr) {
    using T = typename ::std::remove_reference_t<decltype(arr)>::value_type;
    arr.copy_to_host(reinterpret_cast<T*>(host_ptr));
  });
}

void sa_sync(const sa_handle* sa)
{
  dispatch(sa, [](const auto& arr) {
    arr.sync();
  });
}

// ==== tier-1 algorithms =====================================================

void alg_fill(pg_handle* pg, sa_handle* sa, scalar_arg value)
{
  dispatch(sa, [&](auto& arr) {
    using T = typename ::std::remove_reference_t<decltype(arr)>::value_type;
    shd::fill(pg->group, arr, scalar_as<T>(value));
  });
}

void alg_sequence(pg_handle* pg, sa_handle* sa, scalar_arg start, scalar_arg step)
{
  dispatch(sa, [&](auto& arr) {
    using T = typename ::std::remove_reference_t<decltype(arr)>::value_type;
    shd::sequence(pg->group, arr, scalar_as<T>(start), scalar_as<T>(step));
  });
}

namespace
{
template <typename T>
T reduce_typed(place_group& group, const shd::sharded_array<T>& arr, int op)
{
  switch (op)
  {
    case reduce_sum:
      return shd::sum(group, arr);
    case reduce_min:
      return shd::min(group, arr);
    case reduce_max:
      return shd::max(group, arr);
    default:
      throw ::std::invalid_argument("reduce: unknown op descriptor " + ::std::to_string(op));
  }
}
} // namespace

double alg_reduce_f(pg_handle* pg, const sa_handle* sa, int op)
{
  return dispatch(sa, [&](const auto& arr) -> double {
    return static_cast<double>(reduce_typed(pg->group, arr, op));
  });
}

::std::int64_t alg_reduce_i(pg_handle* pg, const sa_handle* sa, int op)
{
  return dispatch(sa, [&](const auto& arr) -> ::std::int64_t {
    return static_cast<::std::int64_t>(reduce_typed(pg->group, arr, op));
  });
}

void alg_inclusive_scan(pg_handle* pg, sa_handle* sa)
{
  dispatch(sa, [&](auto& arr) {
    shd::inclusive_scan(pg->group, arr);
  });
}

void alg_exclusive_scan(pg_handle* pg, sa_handle* sa, scalar_arg init)
{
  dispatch(sa, [&](auto& arr) {
    using T = typename ::std::remove_reference_t<decltype(arr)>::value_type;
    shd::exclusive_scan(pg->group, arr, scalar_as<T>(init));
  });
}

// Helper: run a callable over two same-dtype handles' arrays with the
// concrete array type (dtype equality must already have been checked).
namespace
{
template <typename array_t>
array_t& slot_of(sa_handle* h)
{
  using T = typename array_t::value_type;
  if constexpr (::std::is_same_v<T, float>)
  {
    return h->a_f32;
  }
  else if constexpr (::std::is_same_v<T, double>)
  {
    return h->a_f64;
  }
  else if constexpr (::std::is_same_v<T, ::std::int32_t>)
  {
    return h->a_i32;
  }
  else
  {
    return h->a_i64;
  }
}

template <typename Fn>
void dispatch_pair(sa_handle* a, sa_handle* b, Fn&& fn)
{
  dispatch(a, [&](auto& arr_a) {
    using array_t = ::std::remove_reference_t<decltype(arr_a)>;
    fn(arr_a, slot_of<array_t>(b));
  });
}
} // namespace

void alg_adjacent_difference(pg_handle* pg, sa_handle* input, sa_handle* output)
{
  check_same_dtype(input, output, "adjacent_difference");
  dispatch_pair(input, output, [&](auto& in_arr, auto& out_arr) {
    shd::adjacent_difference(pg->group, in_arr, out_arr);
  });
}

::std::size_t alg_count(pg_handle* pg, const sa_handle* sa, scalar_arg value)
{
  return dispatch(sa, [&](const auto& arr) -> ::std::size_t {
    using T = typename ::std::remove_reference_t<decltype(arr)>::value_type;
    return shd::count(pg->group, arr, scalar_as<T>(value));
  });
}

::std::vector<::std::size_t>
alg_histogram_even(pg_handle* pg, const sa_handle* sa, int num_bins, double lower, double upper)
{
  return dispatch(sa, [&](const auto& arr) {
    using T = typename ::std::remove_reference_t<decltype(arr)>::value_type;
    // Levels are passed in the sample type: bin arithmetic then happens in a
    // type the samples convert to exactly.
    return shd::histogram_even(pg->group, arr, num_bins, static_cast<T>(lower), static_cast<T>(upper));
  });
}

// ==== tier-2, first rung: transform descriptors =============================
//
// The descriptor bodies live in named function templates: nvcc requires an
// extended __device__ lambda's enclosing function to be named and
// address-takeable (a generic lambda would not qualify).

namespace
{
template <typename T>
void transform_unary_typed(
  place_group& group, const shd::sharded_array<T>& input, shd::sharded_array<T>& output, int op, T a)
{
  switch (op)
  {
    case unary_negate:
      shd::transform(group, input, output, [] __device__(T x) {
        return static_cast<T>(-x);
      });
      break;
    case unary_scale:
      shd::transform(group, input, output, [a] __device__(T x) {
        return static_cast<T>(a * x);
      });
      break;
    case unary_add_scalar:
      shd::transform(group, input, output, [a] __device__(T x) {
        return static_cast<T>(x + a);
      });
      break;
    default:
      throw ::std::invalid_argument("transform: unknown unary op descriptor " + ::std::to_string(op));
  }
}

template <typename T>
void transform_binary_typed(
  place_group& group,
  const shd::sharded_array<T>& input1,
  const shd::sharded_array<T>& input2,
  shd::sharded_array<T>& output,
  int op,
  T a)
{
  switch (op)
  {
    case binary_add:
      shd::transform(group, input1, input2, output, [] __device__(T x, T y) {
        return static_cast<T>(x + y);
      });
      break;
    case binary_mul:
      shd::transform(group, input1, input2, output, [] __device__(T x, T y) {
        return static_cast<T>(x * y);
      });
      break;
    case binary_axpy:
      shd::transform(group, input1, input2, output, [a] __device__(T x, T y) {
        return static_cast<T>(a * x + y);
      });
      break;
    default:
      throw ::std::invalid_argument("transform: unknown binary op descriptor " + ::std::to_string(op));
  }
}
} // namespace

void alg_transform_unary(pg_handle* pg, const sa_handle* input, sa_handle* output, int op, scalar_arg alpha)
{
  check_same_dtype(input, output, "transform");
  dispatch_pair(const_cast<sa_handle*>(input), output, [&](auto& in_arr, auto& out_arr) {
    using T = typename ::std::remove_reference_t<decltype(in_arr)>::value_type;
    transform_unary_typed<T>(pg->group, in_arr, out_arr, op, scalar_as<T>(alpha));
  });
}

void alg_transform_binary(
  pg_handle* pg, const sa_handle* input1, const sa_handle* input2, sa_handle* output, int op, scalar_arg alpha)
{
  check_same_dtype(input1, input2, "transform (binary)");
  check_same_dtype(input1, output, "transform (binary)");
  dispatch_pair(const_cast<sa_handle*>(input1), const_cast<sa_handle*>(input2), [&](auto& in1, auto& in2) {
    using T       = typename ::std::remove_reference_t<decltype(in1)>::value_type;
    using array_t = ::std::remove_reference_t<decltype(in1)>;
    transform_binary_typed<T>(pg->group, in1, in2, slot_of<array_t>(output), op, scalar_as<T>(alpha));
  });
}
} // namespace cuda_sharded_shim
