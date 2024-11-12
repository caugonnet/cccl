//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_destructible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__type_traits/remove_all_extents.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// is_nothrow_destructible

#if defined(_CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_DESTRUCTIBLE_FALLBACK)

template <class _Tp>
struct is_nothrow_destructible : public integral_constant<bool, _CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE(_Tp)>
{};

#elif !defined(_LIBCUDACXX_HAS_NO_NOEXCEPT)

template <class _Tp, bool = is_destructible<_Tp>::value>
struct __libcpp_is_nothrow_destructible : false_type
{};

template <class _Tp>
struct __libcpp_is_nothrow_destructible<_Tp, true>
    : public integral_constant<bool, noexcept(_CUDA_VSTD::declval<_Tp>().~_Tp())>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible : public __libcpp_is_nothrow_destructible<_Tp>
{};

template <class _Tp, size_t _Ns>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible<_Tp[_Ns]> : public is_nothrow_destructible<_Tp>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible<_Tp&> : public true_type
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible<_Tp&&> : public true_type
{};

#else

template <class _Tp>
struct __libcpp_nothrow_destructor : public integral_constant<bool, is_scalar<_Tp>::value || is_reference<_Tp>::value>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible
    : public __libcpp_nothrow_destructor<remove_all_extents_t<_Tp>>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_destructible<_Tp[]> : public false_type
{};

#endif // defined(_CCCL_BUILTIN_IS_NOTHROW_DESTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_DESTRUCTIBLE_FALLBACK)

#if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_nothrow_destructible_v = is_nothrow_destructible<_Tp>::value;
#endif // !_CCCL_NO_VARIABLE_TEMPLATES

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DESTRUCTIBLE_H
