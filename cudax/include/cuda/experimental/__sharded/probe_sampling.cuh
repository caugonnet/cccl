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
 * @brief Probe-sampling strategies for splitter selection (internal detail).
 *
 * Sampling-based distribution phases (approximate splitter selection for
 * distribution sorts and rebalancing) draw k probe keys from a SORTED
 * population interval, emitted in population order, in O(k) work. Two
 * strategies with different exactness/scratch trade-offs live here so the
 * engines this tier owns can choose, and so the two can be compared like
 * for like:
 *
 * - `__floyd_sample_n`: Floyd's algorithm — exactly uniform over the
 *   k-subsets (every subset equally likely, every element reachable),
 *   without replacement. Needs a k-element index scratch span.
 * - `__systematic_sample_n`: strided selection with a random phase — O(1)
 *   scratch and a single pass, but not exactly uniform: with stride =
 *   len / k, the trailing `len % k` elements of the population can never
 *   be drawn. Acceptable when k << len and the tail is statistically
 *   unremarkable; kept selectable for comparability with engines that
 *   sample this way.
 *
 * The default is Floyd (exact uniformity, and measured better-or-equal for
 * splitter selection on this tier's workloads). Selection is an INTERNAL
 * experiment knob — `__default_probe_sampler()` honors the
 * `CUDAX_SHARDED_PROBE_SAMPLER` environment variable (`floyd` |
 * `systematic`) — never a public API parameter: call environments carry
 * call semantics, not engine tuning. Engines bound from other layers keep
 * their own samplers; this module only governs sampling phases implemented
 * in this tier.
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

#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__random/uniform_int_distribution.h>
#include <cuda/std/span>

#include <cstdlib>
#include <cstring>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::sharded::reserved
{

//! @brief The probe-sampling strategy selector (internal).
enum class __probe_sampler
{
  __floyd, //!< Floyd's algorithm: exactly uniform without replacement; k-index scratch
  __systematic //!< strided with random phase: O(1) scratch; tail of len % k never drawn
};

//! @brief The process-wide default strategy: Floyd, unless the
//! `CUDAX_SHARDED_PROBE_SAMPLER` environment variable selects otherwise
//! (values: `floyd`, `systematic`). Host-side, read per call (cheap; callers
//! sample once per distribution phase).
inline __probe_sampler __default_probe_sampler()
{
  if (const char* __v = ::std::getenv("CUDAX_SHARDED_PROBE_SAMPLER"))
  {
    if (::std::strcmp(__v, "systematic") == 0)
    {
      return __probe_sampler::__systematic;
    }
  }
  return __probe_sampler::__floyd;
}

//! @brief Systematic sampling: emit up to @p __n elements of
//! [@p __pop_begin, @p __pop_end) at stride len / n from a random phase.
//! O(1) scratch, single pass, output in population order. NOT exactly
//! uniform: the trailing `len % n` population elements are unreachable.
template <class _InputIter, class _OutputIter, class _SizeT, class _Rng>
_CCCL_DEVICE_API _OutputIter
__systematic_sample_n(_InputIter __pop_begin, _InputIter __pop_end, _OutputIter __dest, _SizeT __n, _Rng& __gen)
{
  if (const auto __len = static_cast<_SizeT>(__pop_end - __pop_begin); (__len > 0) && (__n > 0))
  {
    if (__n >= __len)
    {
      for (_SizeT __i = 0; __i < __len; ++__i)
      {
        *__dest++ = __pop_begin[__i];
      }
      return __dest;
    }
    const auto __stride = __len / __n;
    auto __dist         = ::cuda::std::uniform_int_distribution<_SizeT>{0, static_cast<_SizeT>(__stride - 1)};
    const auto __start  = __dist(__gen);
    const auto __stop   = __start + (__n * __stride);
    for (auto __i = __start; __i < __stop; __i += __stride)
    {
      *__dest++ = __pop_begin[__i];
    }
  }
  return __dest;
}

//! @brief Floyd's sampling: emit min(@p __n, len) elements of
//! [@p __pop_begin, @p __pop_end), uniformly without replacement over the
//! k-subsets, in population order. O(k) draws; @p __idx_scratch must hold at
//! least @p __n indices (the chosen set, maintained sorted — insertion is
//! O(k) per draw, fine for the few-hundred-probe regime this serves).
template <class _InputIter, class _OutputIter, class _SizeT, class _Rng>
_CCCL_DEVICE_API _OutputIter __floyd_sample_n(
  _InputIter __pop_begin,
  _InputIter __pop_end,
  _OutputIter __dest,
  _SizeT __n,
  _Rng& __gen,
  ::cuda::std::span<_SizeT> __idx_scratch)
{
  const auto __len = static_cast<_SizeT>(__pop_end - __pop_begin);
  if (__len == 0 || __n == 0)
  {
    return __dest;
  }
  if (__n >= __len)
  {
    for (_SizeT __i = 0; __i < __len; ++__i)
    {
      *__dest++ = __pop_begin[__i];
    }
    return __dest;
  }

  // Floyd: for j = len - n .. len - 1, draw t uniform in [0, j]; take t
  // unless already chosen, else take j. The chosen set stays sorted in
  // __idx_scratch so membership is a lower_bound and emission is a scan.
  _SizeT __count = 0;
  for (_SizeT __j = __len - __n; __j < __len; ++__j)
  {
    auto __dist    = ::cuda::std::uniform_int_distribution<_SizeT>{0, __j};
    const auto __t = __dist(__gen);

    auto* const __pos_it =
      ::cuda::std::lower_bound(__idx_scratch.data(), __idx_scratch.data() + __count, __t);
    const auto __pick = (__pos_it != __idx_scratch.data() + __count && *__pos_it == __t) ? __j : __t;

    // Insert __pick keeping the set sorted (it is not present: __j is new
    // each iteration, and __t was only picked when absent).
    auto* const __ins_it =
      ::cuda::std::lower_bound(__idx_scratch.data(), __idx_scratch.data() + __count, __pick);
    for (auto* __p = __idx_scratch.data() + __count; __p != __ins_it; --__p)
    {
      *__p = *(__p - 1);
    }
    *__ins_it = __pick;
    ++__count;
  }
  for (_SizeT __i = 0; __i < __count; ++__i)
  {
    *__dest++ = __pop_begin[__idx_scratch[__i]];
  }
  return __dest;
}

//! @brief Dispatch on the strategy selector. @p __idx_scratch is consumed
//! only by the Floyd path (systematic ignores it).
template <class _InputIter, class _OutputIter, class _SizeT, class _Rng>
_CCCL_DEVICE_API _OutputIter __probe_sample_n(
  __probe_sampler __which,
  _InputIter __pop_begin,
  _InputIter __pop_end,
  _OutputIter __dest,
  _SizeT __n,
  _Rng& __gen,
  ::cuda::std::span<_SizeT> __idx_scratch)
{
  return __which == __probe_sampler::__floyd
         ? __floyd_sample_n(__pop_begin, __pop_end, __dest, __n, __gen, __idx_scratch)
         : __systematic_sample_n(__pop_begin, __pop_end, __dest, __n, __gen);
}

} // namespace cuda::experimental::sharded::reserved

// NOLINTEND(bugprone-reserved-identifier)
