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
 * @brief C++ shim over the `cuda::experimental::sharded` headers for the
 *        Python bindings.
 *
 * This header is deliberately CUDA-free (plain C++ types only) so the
 * Cython-generated translation unit compiles with the host C++ compiler; the
 * implementations live in .cu files compiled by nvcc, where the header
 * templates are explicitly instantiated for the supported dtypes.
 *
 * Design: every algorithm entry point is ONE Python -> C++ crossing; the C++
 * side owns the per-shard loop, the per-place streams, and the cross-place
 * combine. Errors surface as C++ exceptions (std::invalid_argument for
 * contract violations), which Cython translates to Python exceptions.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cuda_sharded_shim
{
// dtype tags for the explicitly instantiated element types
inline constexpr int dtype_f32 = 0;
inline constexpr int dtype_f64 = 1;
inline constexpr int dtype_i32 = 2;
inline constexpr int dtype_i64 = 3;

// reduce descriptors
inline constexpr int reduce_sum = 0;
inline constexpr int reduce_min = 1;
inline constexpr int reduce_max = 2;

// unary transform descriptors
inline constexpr int unary_negate     = 0; // out = -x
inline constexpr int unary_scale      = 1; // out = alpha * x
inline constexpr int unary_add_scalar = 2; // out = x + alpha

// binary transform descriptors
inline constexpr int binary_add  = 0; // out = x + y
inline constexpr int binary_mul  = 1; // out = x * y
inline constexpr int binary_axpy = 2; // out = alpha * x + y

// Scalar carrier: Python passes both views of a scalar; the shim picks the
// one matching the array's dtype (so 64-bit integers do not round-trip
// through double).
struct scalar_arg
{
  double d;
  ::std::int64_t i;
};

struct pg_handle; // wraps cuda::experimental::places::place_group
struct sa_handle; // wraps sharded_array<T> + dtype tag

// ==== place_group ===========================================================

pg_handle* pg_by_locality_domains(const ::std::vector<int>& device_ids);
pg_handle* pg_by_devices(const ::std::vector<int>& device_ids);
void pg_destroy(pg_handle* pg) noexcept;
::std::size_t pg_size(const pg_handle* pg);
::std::uintptr_t pg_get_stream(pg_handle* pg, ::std::size_t place_idx, ::std::size_t color);
void pg_sync(pg_handle* pg);

// ==== sharded_array =========================================================

sa_handle* sa_allocate(pg_handle* pg, int dtype, ::std::size_t total_size);
sa_handle* sa_allocate_sizes(pg_handle* pg, int dtype, const ::std::vector<::std::size_t>& sizes);
sa_handle* sa_allocate_contiguous(pg_handle* pg, int dtype, ::std::size_t total_size);

// Adopt externally owned per-shard device buffers (non-owning view). Buffer i
// is treated as resident at group place i; `producer_streams[i]`, when
// non-zero, is synchronized before the buffers are used (the consumer-waits
// rule of the CUDA array interface).
sa_handle* sa_adopt(pg_handle* pg,
                    int dtype,
                    const ::std::vector<::std::uintptr_t>& ptrs,
                    const ::std::vector<::std::size_t>& sizes,
                    const ::std::vector<::std::uintptr_t>& producer_streams);

void sa_destroy(sa_handle* sa) noexcept;
int sa_dtype(const sa_handle* sa);
::std::size_t sa_size(const sa_handle* sa);
::std::size_t sa_num_shards(const sa_handle* sa);
bool sa_is_contiguous(const sa_handle* sa);
::std::uintptr_t sa_contiguous_ptr(const sa_handle* sa);
void sa_shard_info(
  const sa_handle* sa,
  ::std::size_t idx,
  ::std::uintptr_t* data,
  ::std::size_t* size,
  ::std::size_t* global_offset,
  ::std::uintptr_t* stream);
void sa_copy_from_host(sa_handle* sa, ::std::uintptr_t host_ptr); // synchronous
void sa_copy_to_host(const sa_handle* sa, ::std::uintptr_t host_ptr); // synchronous
void sa_sync(const sa_handle* sa);

// ==== tier-1 algorithms (opaque: one crossing, C++ owns the loop) ===========

void alg_fill(pg_handle* pg, sa_handle* sa, scalar_arg value);
void alg_sequence(pg_handle* pg, sa_handle* sa, scalar_arg start, scalar_arg step);
double alg_reduce_f(pg_handle* pg, const sa_handle* sa, int op); // f32/f64
::std::int64_t alg_reduce_i(pg_handle* pg, const sa_handle* sa, int op); // i32/i64
void alg_inclusive_scan(pg_handle* pg, sa_handle* sa); // sum, in place
void alg_exclusive_scan(pg_handle* pg, sa_handle* sa, scalar_arg init); // sum, in place
void alg_adjacent_difference(pg_handle* pg, sa_handle* input, sa_handle* output);
::std::size_t alg_count(pg_handle* pg, const sa_handle* sa, scalar_arg value);
::std::vector<::std::size_t>
alg_histogram_even(pg_handle* pg, const sa_handle* sa, int num_bins, double lower, double upper);
void alg_sort(pg_handle* pg, sa_handle* sa); // ascending, in place

// ==== tier-2, first rung: transform with standard-op descriptors ============

void alg_transform_unary(pg_handle* pg, const sa_handle* input, sa_handle* output, int op, scalar_arg alpha);
void alg_transform_binary(
  pg_handle* pg, const sa_handle* input1, const sa_handle* input2, sa_handle* output, int op, scalar_arg alpha);
} // namespace cuda_sharded_shim
