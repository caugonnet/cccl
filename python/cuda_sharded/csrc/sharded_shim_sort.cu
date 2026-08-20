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
 * @brief Shim implementation of sort, in its own translation unit: it
 *        instantiates the distributed sort engine for each dtype, which
 *        dominates compile time.
 */

#include "sharded_shim_impl.h"

namespace cuda_sharded_shim
{
void alg_sort(pg_handle* pg, sa_handle* sa)
{
  dispatch(sa, [&](auto& arr) {
    shd::sort(pg->group, arr);
  });
}
} // namespace cuda_sharded_shim
