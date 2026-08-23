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
 *
 * @brief Correctness of the sharded reduce / scan / adjacent_difference
 *        algorithms against host references, over multiple places: per-place
 *        CUB primitive plus cross-place combine.
 */

#include <cuda/experimental/sharded.cuh>

#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::place_group;

namespace
{
struct max_op
{
  __host__ __device__ long long operator()(long long a, long long b) const
  {
    return a > b ? a : b;
  }
};

void test_reduce(place_group& group)
{
  const size_t n = 1000001;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(group, data, 1LL); // 1..n

  const long long expected_sum = static_cast<long long>(n) * (static_cast<long long>(n) + 1) / 2;
  EXPECT(sum(group, data) == expected_sum);
  EXPECT(min(group, data) == 1LL);
  EXPECT(max(group, data) == static_cast<long long>(n));

  // Custom operator through the generic entry point
  EXPECT(reduce(group, data, max_op{}, 0LL) == static_cast<long long>(n));

  // Empty array returns the initial value
  sharded_array<long long> empty;
  EXPECT(reduce(group, empty, max_op{}, -7LL) == -7LL);
}

void test_inclusive_scan(place_group& group)
{
  const size_t n = 262147;
  auto data      = sharded_array<long long>::allocate(group, n);
  fill(group, data, 1LL);

  inclusive_scan(group, data); // 1, 2, 3, ..., n
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<long long>(i) + 1);
  }

  // Custom operator: running maximum of iota is iota itself
  iota(group, data, 0LL);
  inclusive_scan(group, data, max_op{});
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<long long>(i));
  }
}

void test_exclusive_scan(place_group& group)
{
  const size_t n = 131075;
  auto data      = sharded_array<long long>::allocate(group, n);
  fill(group, data, 2LL);

  exclusive_scan(group, data); // 0, 2, 4, ..., 2*(n-1)
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 2 * static_cast<long long>(i));
  }

  // inclusive_sum / exclusive_sum aliases
  fill(group, data, 1LL);
  inclusive_sum(group, data);
  data.copy_to_host(host.data());
  EXPECT(host[n - 1] == static_cast<long long>(n));

  fill(group, data, 1LL);
  exclusive_sum(group, data);
  data.copy_to_host(host.data());
  EXPECT(host[0] == 0LL);
  EXPECT(host[n - 1] == static_cast<long long>(n) - 1);
}

void test_adjacent_difference(place_group& group)
{
  const size_t n = 100000;
  auto input     = sharded_array<long long>::allocate(group, n);
  auto output    = sharded_array<long long>::allocate_like(input);

  // input[i] = i*i; diff[i] = i*i - (i-1)^2 = 2i - 1 (and diff[0] = 0)
  ::std::vector<long long> host(n);
  for (size_t i = 0; i < n; i++)
  {
    host[i] = static_cast<long long>(i) * static_cast<long long>(i);
  }
  input.copy_from_host(host.data());

  adjacent_difference(group, input, output);
  output.copy_to_host(host.data());

  EXPECT(host[0] == 0LL); // first element kept as-is
  for (size_t i = 1; i < n; i++)
  {
    if (host[i] != 2 * static_cast<long long>(i) - 1)
    {
    }
    EXPECT(host[i] == 2 * static_cast<long long>(i) - 1);
  }

  // The cross-shard boundary elements are exercised whenever the group has
  // more than one place (indices at shard boundaries take the prev_last path)
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();

  test_reduce(group);
  test_inclusive_scan(group);
  test_exclusive_scan(group);
  test_adjacent_difference(group);

  return 0;
}
