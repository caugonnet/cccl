//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_DEDUCTION_GUIDES_SFINAE_CHECKS_H
#define TEST_SUPPORT_DEDUCTION_GUIDES_SFINAE_CHECKS_H

#include <cuda/std/__memory_>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/initializer_list>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// `SFINAEs_away` template variable checks whether the template arguments for
// a given template class `Instantiated` can be deduced from the given
// constructor parameter types `CtrArgs` using CTAD.

template <template <typename...> class Instantiated,
          class... CtrArgs,
          class = decltype(Instantiated(cuda::std::declval<CtrArgs>()...))>
__host__ __device__ cuda::std::false_type SFINAEs_away_impl(int);

template <template <typename...> class Instantiated, class... CtrArgs>
__host__ __device__ cuda::std::true_type SFINAEs_away_impl(...);

template <template <typename...> class Instantiated, class... CtrArgs>
constexpr bool SFINAEs_away = decltype(SFINAEs_away_impl<Instantiated, CtrArgs...>(0))::value;

// For sequence containers the deduction guides should be SFINAE'd away when
// given:
// - "bad" input iterators (that is, a type not qualifying as an input
//   iterator);
// - a bad allocator.
template <template <typename...> class Container, typename InstantiatedContainer>
__host__ __device__ constexpr void SequenceContainerDeductionGuidesSfinaeAway()
{
  using Alloc = cuda::std::allocator<int>;
  using Iter  = int*;

  struct BadAlloc
  {};
  // Note: the only requirement in the Standard is that integral types cannot be
  // considered input iterators; however, this doesn't work for sequence
  // containers because they have constructors of the form `(size_type count,
  // const value_type& value)`. These constructors would be used when passing
  // two integral types and would deduce `value_type` to be an integral type.
#ifdef _LIBCUDACXX_VERSION
  using OutputIter = cuda::std::insert_iterator<InstantiatedContainer>;
#endif // _LIBCUDACXX_VERSION

  // (iter, iter)
  //
  // Cannot deduce from (BAD_iter, BAD_iter)
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter>);

  // (iter, iter, alloc)
  //
  // Cannot deduce from (BAD_iter, BAD_iter, alloc)
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter, Alloc>);
  // Cannot deduce from (iter, iter, BAD_alloc)
  static_assert(SFINAEs_away<Container, Iter, Iter, BadAlloc>);

  // (alloc)
  //
  // Cannot deduce from (alloc)
  static_assert(SFINAEs_away<Container, Alloc>);
}

// For associative containers the deduction guides should be SFINAE'd away when
// given:
// - "bad" input iterators (that is, a type not qualifying as an input
//   iterator);
// - a bad allocator;
// - an allocator in place of a comparator.

template <template <typename...> class Container, typename InstantiatedContainer>
__host__ __device__ constexpr void AssociativeContainerDeductionGuidesSfinaeAway()
{
  using ValueType = typename InstantiatedContainer::value_type;
  using Comp      = cuda::std::less<int>;
  using Alloc     = cuda::std::allocator<ValueType>;
  using Iter      = ValueType*;
  using InitList  = cuda::std::initializer_list<ValueType>;

  struct BadAlloc
  {};
  // The only requirement in the Standard is that integral types cannot be
  // considered input iterators, beyond that it is unspecified.
  using BadIter = int;
#ifdef _LIBCUDACXX_VERSION
  using OutputIter = cuda::std::insert_iterator<InstantiatedContainer>;
#endif // _LIBCUDACXX_VERSION
  using AllocAsComp = Alloc;

  // (iter, iter)
  //
  // Cannot deduce from (BAD_iter, BAD_iter)
  static_assert(SFINAEs_away<Container, BadIter, BadIter>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter>);

  // (iter, iter, comp)
  //
  // Cannot deduce from (BAD_iter, BAD_iter, comp)
  static_assert(SFINAEs_away<Container, BadIter, BadIter, Comp>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter, Comp>);

  // (iter, iter, comp, alloc)
  //
  // Cannot deduce from (BAD_iter, BAD_iter, comp, alloc)
  static_assert(SFINAEs_away<Container, BadIter, BadIter, Comp, Alloc>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter, Comp, Alloc>);
  // Cannot deduce from (iter, iter, ALLOC_as_comp, alloc)
  static_assert(SFINAEs_away<Container, Iter, Iter, AllocAsComp, Alloc>);
  // Cannot deduce from (iter, iter, comp, BAD_alloc)
  static_assert(SFINAEs_away<Container, Iter, Iter, Comp, BadAlloc>);

  // (iter, iter, alloc)
  //
  // Cannot deduce from (BAD_iter, BAD_iter, alloc)
  static_assert(SFINAEs_away<Container, BadIter, BadIter, Alloc>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter, Alloc>);
  // Note: (iter, iter, BAD_alloc) is interpreted as (iter, iter, comp)
  // instead and fails upon instantiation. There is no requirement to SFINAE
  // away bad comparators.

  // (init_list, comp, alloc)
  //
  // Cannot deduce from (init_list, ALLOC_as_comp, alloc)
  static_assert(SFINAEs_away<Container, InitList, AllocAsComp, Alloc>);
  // Cannot deduce from (init_list, comp, BAD_alloc)
  static_assert(SFINAEs_away<Container, InitList, Comp, BadAlloc>);

  // (init_list, alloc)
  //
  // Note: (init_list, BAD_alloc) is interpreted as (init_list, comp) instead
  // and fails upon instantiation. There is no requirement to SFINAE away bad
  // comparators.
}

// For unordered containers the deduction guides should be SFINAE'd away when
// given:
// - "bad" input iterators (that is, a type not qualifying as an input
//   iterator);
// - a bad allocator;
// - a bad hash functor (an integral type in place of a hash);
// - an allocator in place of a hash functor;
// - an allocator in place of a predicate.
template <template <typename...> class Container, typename InstantiatedContainer>
__host__ __device__ constexpr void UnorderedContainerDeductionGuidesSfinaeAway()
{
  using ValueType = typename InstantiatedContainer::value_type;
  using Pred      = cuda::std::equal_to<int>;
  using Hash      = cuda::std::hash<int>;
  using Alloc     = cuda::std::allocator<ValueType>;
  using Iter      = ValueType*;
  using InitList  = cuda::std::initializer_list<ValueType>;

  using BadHash = int;
  struct BadAlloc
  {};
  // The only requirement in the Standard is that integral types cannot be
  // considered input iterators, beyond that it is unspecified.
  using BadIter = int;
#ifdef _LIBCUDACXX_VERSION
  using OutputIter = cuda::std::insert_iterator<InstantiatedContainer>;
#endif // _LIBCUDACXX_VERSION
  using AllocAsHash = Alloc;
  using AllocAsPred = Alloc;

  // (iter, iter)
  //
  // Cannot deduce from (BAD_iter, BAD_iter)
  static_assert(SFINAEs_away<Container, BadIter, BadIter>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter>);

  // (iter, iter, buckets)
  //
  // Cannot deduce from (BAD_iter, BAD_iter, buckets)
  static_assert(SFINAEs_away<Container, BadIter, BadIter, cuda::std::size_t>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter, cuda::std::size_t>);

  // (iter, iter, buckets, hash)
  //
  // Cannot deduce from (BAD_iter, BAD_iter, buckets, hash)
  static_assert(SFINAEs_away<Container, BadIter, BadIter, cuda::std::size_t, Hash>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter, cuda::std::size_t, Hash>);
  // Cannot deduce from (iter, iter, buckets, BAD_hash)
  static_assert(SFINAEs_away<Container, Iter, Iter, cuda::std::size_t, BadHash>);
  // Note: (iter, iter, buckets, ALLOC_as_hash) is allowed -- it just calls
  // (iter, iter, buckets, alloc)

  // (iter, iter, buckets, hash, pred)
  //
  // Cannot deduce from (BAD_iter, BAD_iter, buckets, hash, pred)
  static_assert(SFINAEs_away<Container, BadIter, BadIter, cuda::std::size_t, Hash, Pred>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter, cuda::std::size_t, Hash, Pred>);
  // Cannot deduce from (iter, iter, buckets, BAD_hash, pred)
  static_assert(SFINAEs_away<Container, Iter, Iter, cuda::std::size_t, BadHash, Pred>);
  // Cannot deduce from (iter, iter, buckets, ALLOC_as_hash, pred)
  static_assert(SFINAEs_away<Container, Iter, Iter, cuda::std::size_t, AllocAsHash, Pred>);
  // Note: (iter, iter, buckets, hash, ALLOC_as_pred) is allowed -- it just
  // calls (iter, iter, buckets, hash, alloc)

  // (iter, iter, buckets, hash, pred, alloc)
  //
  // Cannot deduce from (BAD_iter, BAD_iter, buckets, hash, pred, alloc)
  static_assert(SFINAEs_away<Container, BadIter, BadIter, cuda::std::size_t, Hash, Pred, Alloc>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter, cuda::std::size_t, Hash, Pred, Alloc>);
  // Cannot deduce from (iter, iter, buckets, BAD_hash, pred, alloc)
  static_assert(SFINAEs_away<Container, Iter, Iter, cuda::std::size_t, BadHash, Pred, Alloc>);
  // Cannot deduce from (iter, iter, buckets, ALLOC_as_hash, pred, alloc)
  static_assert(SFINAEs_away<Container, Iter, Iter, cuda::std::size_t, AllocAsHash, Pred, Alloc>);
  // Cannot deduce from (iter, iter, buckets, hash, ALLOC_as_pred, alloc)
  static_assert(SFINAEs_away<Container, Iter, Iter, cuda::std::size_t, Hash, AllocAsPred, Alloc>);
  // Cannot deduce from (iter, iter, buckets, hash, pred, BAD_alloc)
  static_assert(SFINAEs_away<Container, Iter, Iter, cuda::std::size_t, Hash, Pred, BadAlloc>);

  // (iter, iter, buckets, alloc)
  //
  // Cannot deduce from (BAD_iter, BAD_iter, buckets, alloc)
  static_assert(SFINAEs_away<Container, BadIter, BadIter, cuda::std::size_t, Alloc>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter, cuda::std::size_t, Alloc>);
  // Note: (iter, iter, buckets, BAD_alloc) is interpreted as (iter, iter,
  // buckets, hash), which is valid because the only requirement for the hash
  // parameter is that it's not integral.

  // (iter, iter, alloc)
  //
  // Cannot deduce from (BAD_iter, BAD_iter, alloc)
  static_assert(SFINAEs_away<Container, BadIter, BadIter, Alloc>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter, Alloc>);
  // Cannot deduce from (iter, iter, BAD_alloc)
  static_assert(SFINAEs_away<Container, Iter, Iter, BadAlloc>);

  // (iter, iter, buckets, hash, alloc)
  //
  // Cannot deduce from (BAD_iter, BAD_iter, buckets, hash, alloc)
  static_assert(SFINAEs_away<Container, BadIter, BadIter, cuda::std::size_t, Hash, Alloc>);
  static_assert(SFINAEs_away<Container, OutputIter, OutputIter, cuda::std::size_t, Hash, Alloc>);
  // Cannot deduce from (iter, iter, buckets, BAD_hash, alloc)
  static_assert(SFINAEs_away<Container, Iter, Iter, cuda::std::size_t, BadHash, Alloc>);
  // Cannot deduce from (iter, iter, buckets, ALLOC_as_hash, alloc)
  static_assert(SFINAEs_away<Container, Iter, Iter, cuda::std::size_t, AllocAsHash, Alloc>);
  // Note: (iter, iter, buckets, hash, BAD_alloc) is interpreted as (iter, iter,
  // buckets, hash, pred), which is valid because there are no requirements for
  // the predicate.

  // (init_list, buckets, hash)
  //
  // Cannot deduce from (init_list, buckets, BAD_hash)
  static_assert(SFINAEs_away<Container, InitList, cuda::std::size_t, BadHash>);
  // Note: (init_list, buckets, ALLOC_as_hash) is interpreted as (init_list,
  // buckets, alloc), which is valid.

  // (init_list, buckets, hash, pred)
  //
  // Cannot deduce from (init_list, buckets, BAD_hash, pred)
  static_assert(SFINAEs_away<Container, InitList, cuda::std::size_t, BadHash, Pred>);
  // Cannot deduce from (init_list, buckets, ALLOC_as_hash, pred)
  static_assert(SFINAEs_away<Container, InitList, cuda::std::size_t, AllocAsHash, Pred>);
  // Note: (init_list, buckets, hash, ALLOC_as_pred) is interpreted as
  // (init_list, buckets, hash, alloc), which is valid.

  // (init_list, buckets, hash, pred, alloc)
  //
  // Cannot deduce from (init_list, buckets, BAD_hash, pred, alloc)
  static_assert(SFINAEs_away<Container, InitList, cuda::std::size_t, BadHash, Pred, Alloc>);
  // Cannot deduce from (init_list, buckets, ALLOC_as_hash, pred, alloc)
  static_assert(SFINAEs_away<Container, InitList, cuda::std::size_t, AllocAsHash, Pred, Alloc>);
  // Cannot deduce from (init_list, buckets, hash, ALLOC_as_pred, alloc)
  static_assert(SFINAEs_away<Container, InitList, cuda::std::size_t, Hash, AllocAsPred, Alloc>);
  // Cannot deduce from (init_list, buckets, hash, pred, BAD_alloc)
  static_assert(SFINAEs_away<Container, InitList, cuda::std::size_t, Hash, Pred, BadAlloc>);

  // (init_list, buckets, alloc)
  //
  // Note: (init_list, buckets, BAD_alloc) is interpreted as (init_list,
  // buckets, hash), which is valid because the only requirement for the hash
  // parameter is that it's not integral.

  // (init_list, buckets, hash, alloc)
  //
  // Cannot deduce from (init_list, buckets, BAD_hash, alloc)
  static_assert(SFINAEs_away<Container, InitList, cuda::std::size_t, BadHash, Alloc>);
  // Cannot deduce from (init_list, buckets, ALLOC_as_hash, alloc)
  static_assert(SFINAEs_away<Container, InitList, cuda::std::size_t, AllocAsHash, Alloc>);

  // (init_list, alloc)
  //
  // Cannot deduce from (init_list, BAD_alloc)
  static_assert(SFINAEs_away<Container, InitList, BadAlloc>);
}

#endif // TEST_SUPPORT_DEDUCTION_GUIDES_SFINAE_CHECKS_H
