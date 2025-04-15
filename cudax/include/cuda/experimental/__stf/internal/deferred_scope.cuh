//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Generic class to defer to describe deferred scopes
 *
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

namespace cuda::experimental::stf
{

template <typename real_scope_t>
class deferred_scope {
    deferred_scope(args...) : scope(args...) {}

  auto& set_symbol(::std::string s)
  {
    return scope.set_symbol(mv(s));
  }

  template <typename Fun>
  void operator->*(Fun&& f)
  {
      scope->*::std::forward<Fun>(f);
  }

public:
  real_scope_t scope;
};

} // end namespace cuda::experimental::stf
