//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Main include file for the CUDASTF library.
 */

#pragma once

#include <cuda/experimental/__stf/allocators/adapters.cuh>
#include <cuda/experimental/__stf/allocators/buddy_allocator.cuh>
#include <cuda/experimental/__stf/allocators/cached_allocator.cuh>
#include <cuda/experimental/__stf/allocators/pooled_allocator.cuh>
#include <cuda/experimental/__stf/allocators/uncached_allocator.cuh>
#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/internal/reducer.cuh>
#include <cuda/experimental/__stf/internal/scalar_interface.cuh>
#include <cuda/experimental/__stf/internal/task_dep.cuh>
#include <cuda/experimental/__stf/internal/void_interface.cuh>
#include <cuda/experimental/__stf/places/exec/cuda_stream.cuh>
#include <cuda/experimental/__stf/places/inner_shape.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <map>
#include <variant>

namespace cuda::experimental::stf
{

template <typename underlying_ctx_t>
class deferred_context
{
public:
  deferred_context() = default;

  template <typename... Args>
  decltype(auto) task(Args... args)
  {
    return underlying_ctx.task(::std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) logical_data(Args... args)
  {
    return underlying_ctx.logical_data(::std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) parallel_for(Args... args)
  {
    return underlying_ctx.parallel_for(::std::forward<Args>(args)...);
  }

  void finalize()
  {
    return underlying_ctx.finalize();
  }

private:
  underlying_ctx_t underlying_ctx;
};

} // end namespace cuda::experimental::stf
