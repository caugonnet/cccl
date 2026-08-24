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
 * @brief Internal shim plumbing shared by the .cu translation units: handle
 *        definitions, dtype dispatch, scalar conversion. NOT included by the
 *        Cython TU (this header pulls in the CUDA headers).
 */

#pragma once

#include <cuda/experimental/sharded.cuh>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "sharded_shim.h"

namespace cuda_sharded_shim
{
namespace shd = ::cuda::experimental::sharded;
using ::cuda::experimental::places::place_group;

struct pg_handle
{
  place_group group;
};

// dtype-erased container handle: exactly one of the four members is engaged,
// selected by `dtype`. A plain tagged struct (rather than std::variant) keeps
// the dispatch readable from the .cu files and the error paths explicit.
struct sa_handle
{
  int dtype = -1;
  shd::sharded_array<float> a_f32;
  shd::sharded_array<double> a_f64;
  shd::sharded_array<::std::int32_t> a_i32;
  shd::sharded_array<::std::int64_t> a_i64;
};

// Dispatch a callable over the array matching the handle's dtype tag. This is
// where the header templates get explicitly instantiated for each dtype.
template <typename Fn>
decltype(auto) dispatch(sa_handle* sa, Fn&& fn)
{
  switch (sa->dtype)
  {
    case dtype_f32:
      return fn(sa->a_f32);
    case dtype_f64:
      return fn(sa->a_f64);
    case dtype_i32:
      return fn(sa->a_i32);
    case dtype_i64:
      return fn(sa->a_i64);
    default:
      throw ::std::invalid_argument("cuda_sharded: unknown dtype tag " + ::std::to_string(sa->dtype));
  }
}

template <typename Fn>
decltype(auto) dispatch(const sa_handle* sa, Fn&& fn)
{
  return dispatch(const_cast<sa_handle*>(sa), [&fn](auto& arr) -> decltype(auto) {
    return fn(::std::as_const(arr));
  });
}

// Pick the scalar view matching an element type.
template <typename T>
T scalar_as(const scalar_arg& s)
{
  if constexpr (::std::is_floating_point_v<T>)
  {
    return static_cast<T>(s.d);
  }
  else
  {
    return static_cast<T>(s.i);
  }
}

inline void check_same_dtype(const sa_handle* a, const sa_handle* b, const char* context)
{
  if (a->dtype != b->dtype)
  {
    throw ::std::invalid_argument(::std::string(context) + ": dtype mismatch (" + ::std::to_string(a->dtype) + " vs "
                                  + ::std::to_string(b->dtype) + ")");
  }
}
} // namespace cuda_sharded_shim
