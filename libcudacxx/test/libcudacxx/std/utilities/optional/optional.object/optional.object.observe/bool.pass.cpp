//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// constexpr explicit optional<T>::operator bool() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using cuda::std::optional;
  {
    const optional<int> opt;
    ((void) opt);
    ASSERT_NOEXCEPT(bool(opt));
    static_assert(!cuda::std::is_convertible<optional<int>, bool>::value, "");
  }

  {
    constexpr optional<int> opt;
    static_assert(!opt, "");
  }
  {
    constexpr optional<int> opt(0);
    static_assert(opt, "");
  }

  return 0;
}
