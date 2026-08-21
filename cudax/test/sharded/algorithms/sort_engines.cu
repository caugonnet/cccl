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
 * @brief The tier-2 engine matrix of `sharded::sort`: both engines
 *        (shared-address-space and distributed) against `std::sort` over
 *        mixed distributions and sizes, cross-engine byte agreement,
 *        per-engine bitwise run-to-run identity, boundary/metadata
 *        preservation on even, uneven and contiguous layouts, and the
 *        eligibility rules of the explicit engine request.
 */

#include <cuda/experimental/sharded.cuh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::place_group;

namespace
{
template <typename T>
::std::vector<T> make_dataset(const char* kind, size_t n, unsigned seed)
{
  ::std::vector<T> host(n);
  ::std::mt19937 rng(seed);
  if (::std::strcmp(kind, "uniform") == 0)
  {
    ::std::uniform_real_distribution<double> dist(-1e6, 1e6);
    for (auto& v : host)
    {
      v = static_cast<T>(dist(rng));
    }
  }
  else if (::std::strcmp(kind, "lognormal") == 0)
  {
    ::std::lognormal_distribution<double> dist(0.0, 2.0);
    for (auto& v : host)
    {
      v = static_cast<T>(dist(rng));
    }
  }
  else if (::std::strcmp(kind, "all-equal") == 0)
  {
    ::std::fill(host.begin(), host.end(), static_cast<T>(42));
  }
  else if (::std::strcmp(kind, "pre-sorted") == 0)
  {
    for (size_t i = 0; i < n; i++)
    {
      host[i] = static_cast<T>(i);
    }
  }
  else if (::std::strcmp(kind, "reverse") == 0)
  {
    for (size_t i = 0; i < n; i++)
    {
      host[i] = static_cast<T>(n - i);
    }
  }
  else // dup-heavy
  {
    ::std::uniform_int_distribution<int> dist(0, 15);
    for (auto& v : host)
    {
      v = static_cast<T>(dist(rng));
    }
  }
  return host;
}

// Sort `host` through the given engine on a fresh copy of the data in `data`,
// checking the result against std::sort and the layout against itself.
template <typename T>
::std::vector<T> run_engine(place_group& group, sharded_array<T>& data, const ::std::vector<T>& host, sort_engine engine)
{
  ::std::vector<size_t> sizes_before, offsets_before, caps_before;
  for (size_t g = 0; g < data.num_shards(); g++)
  {
    sizes_before.push_back(data.shard(g).size);
    offsets_before.push_back(data.shard(g).global_offset);
    caps_before.push_back(data.shard(g).capacity);
  }
  const size_t total_before = data.size();

  data.copy_from_host(host.data());
  sort(group, data, ::cuda::std::less<T>{}, engine);

  ::std::vector<T> got(host.size());
  data.copy_to_host(got.data());

  ::std::vector<T> ref = host;
  ::std::sort(ref.begin(), ref.end());
  EXPECT(::std::memcmp(got.data(), ref.data(), ref.size() * sizeof(T)) == 0);

  EXPECT(data.size() == total_before);
  for (size_t g = 0; g < data.num_shards(); g++)
  {
    EXPECT(data.shard(g).size == sizes_before[g]);
    EXPECT(data.shard(g).global_offset == offsets_before[g]);
    EXPECT(data.shard(g).capacity == caps_before[g]);
  }
  EXPECT(data.validate());
  return got;
}

void test_engine_matrix(place_group& group)
{
  const char* kinds[] = {"uniform", "lognormal", "all-equal", "pre-sorted", "reverse", "dup-heavy"};
  const size_t sizes[] = {64, 4097, (1 << 20) + 37};

  for (const size_t n : sizes)
  {
    // Uneven shards: first shard ~2x the others (as long as n allows).
    ::std::vector<size_t> shard_sizes(group.size(), n / (2 * group.size() - 1));
    shard_sizes[0] = n - (group.size() - 1) * (n / (2 * group.size() - 1));

    auto fdata = sharded_array<float>::allocate(group, shard_sizes);
    auto idata = sharded_array<int>::allocate(group, shard_sizes);

    for (const char* kind : kinds)
    {
      {
        const auto host = make_dataset<float>(kind, n, 1234);
        const auto a    = run_engine(group, fdata, host, sort_engine::shared_va);
        const auto b    = run_engine(group, fdata, host, sort_engine::distributed);
        // Keys-only: the sorted multiset is unique, so the two engines must
        // agree byte for byte.
        EXPECT(::std::memcmp(a.data(), b.data(), n * sizeof(float)) == 0);
      }
      {
        const auto host = make_dataset<int>(kind, n, 4321);
        const auto a    = run_engine(group, idata, host, sort_engine::shared_va);
        const auto b    = run_engine(group, idata, host, sort_engine::distributed);
        EXPECT(::std::memcmp(a.data(), b.data(), n * sizeof(int)) == 0);
      }
    }
  }
}

struct descending
{
  __host__ __device__ bool operator()(float a, float b) const
  {
    return a > b;
  }
};

void test_custom_comparator_both(place_group& group)
{
  // A custom comparator takes the comparator-generic local-sort path in the
  // shared_va engine; both engines must still agree with std::sort.
  const size_t n = 300001;
  auto data      = sharded_array<float>::allocate(group, n);
  auto host      = make_dataset<float>("uniform", n, 99);

  for (const auto engine : {sort_engine::shared_va, sort_engine::distributed})
  {
    data.copy_from_host(host.data());
    sort(group, data, descending{}, engine);

    ::std::vector<float> got(n);
    data.copy_to_host(got.data());
    ::std::vector<float> ref = host;
    ::std::sort(ref.begin(), ref.end(), ::std::greater<float>{});
    EXPECT(::std::memcmp(got.data(), ref.data(), n * sizeof(float)) == 0);
  }

  // greater<T> takes the radix descending fast path; same contract.
  data.copy_from_host(host.data());
  sort(group, data, ::cuda::std::greater<float>{}, sort_engine::shared_va);
  ::std::vector<float> got(n);
  data.copy_to_host(got.data());
  ::std::vector<float> ref = host;
  ::std::sort(ref.begin(), ref.end(), ::std::greater<float>{});
  EXPECT(::std::memcmp(got.data(), ref.data(), n * sizeof(float)) == 0);
}

void test_contiguous_both(place_group& group)
{
  // Both engines must land the sorted slices exactly on the fixed boundaries
  // of the contiguous backing: contiguous_data() reads as ONE sorted array.
  const size_t n = 500000;
  auto data      = sharded_array<long long>::allocate_contiguous(group, n);
  EXPECT(data.is_contiguous());

  ::std::vector<long long> host(n);
  ::std::mt19937_64 rng(2026);
  for (auto& v : host)
  {
    v = static_cast<long long>(rng());
  }
  ::std::vector<long long> ref = host;
  ::std::sort(ref.begin(), ref.end());

  for (const auto engine : {sort_engine::shared_va, sort_engine::distributed})
  {
    data.copy_from_host(host.data());
    sort(group, data, ::cuda::std::less<long long>{}, engine);
    EXPECT(data.validate());

    ::std::vector<long long> got(n);
    cuda_safe_call(cudaMemcpy(got.data(), data.contiguous_data(), n * sizeof(long long), cudaMemcpyDefault));
    EXPECT(::std::memcmp(got.data(), ref.data(), n * sizeof(long long)) == 0);
  }
}

void test_bitwise_repeat_both(place_group& group)
{
  const size_t n = 250007;
  auto data      = sharded_array<float>::allocate(group, n);
  const auto host = make_dataset<float>("uniform", n, 5);

  for (const auto engine : {sort_engine::shared_va, sort_engine::distributed})
  {
    ::std::vector<float> first(n), again(n);
    data.copy_from_host(host.data());
    sort(group, data, ::cuda::std::less<float>{}, engine);
    data.copy_to_host(first.data());

    for (int rep = 0; rep < 2; rep++)
    {
      data.copy_from_host(host.data());
      sort(group, data, ::cuda::std::less<float>{}, engine);
      data.copy_to_host(again.data());
      EXPECT(::std::memcmp(again.data(), first.data(), n * sizeof(float)) == 0);
    }
  }
}

void test_eligibility(place_group& group)
{
  // Device-backed shards of one device share an address space.
  {
    auto dev = sharded_array<int>::allocate(group, size_t{1024});
    EXPECT(detail_sort_va::one_shared_address_space(dev));
  }

  // A host shard does not; the explicit shared_va request is refused.
  {
    auto host_arr = sharded_array<int>::allocate_host(1024);
    EXPECT(!detail_sort_va::one_shared_address_space(host_arr));

    // Give it a matching single-place "group" to pass the shape check.
    place_group host_group(::std::vector<cuda::experimental::places::exec_place>{
      cuda::experimental::places::exec_place::host()});
    bool threw = false;
    try
    {
      sort(host_group, host_arr, ::cuda::std::less<int>{}, sort_engine::shared_va);
    }
    catch (const ::std::invalid_argument&)
    {
      threw = true;
    }
    EXPECT(threw);
  }
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();

  test_engine_matrix(group);
  test_custom_comparator_both(group);
  test_contiguous_both(group);
  test_bitwise_repeat_both(group);
  test_eligibility(group);

  return 0;
}
