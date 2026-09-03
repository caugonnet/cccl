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
 * @brief The probe-sampling strategies: output-shape contracts (count,
 *        population order, distinctness, whole-population clamp), the
 *        statistical distinction the two strategies are DOCUMENTED to have
 *        (Floyd reaches every population element with near-uniform
 *        frequency; systematic never draws the trailing len % n elements),
 *        and the default/env selection.
 */

#include <cuda/experimental/__sharded/probe_sampling.cuh>

#include <cuda/std/__random/philox_engine.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

// Standalone EXPECT (this test includes only the sampling header).
#define EXPECT(cond)                                                                    \
  do                                                                                    \
  {                                                                                     \
    if (!(cond))                                                                        \
    {                                                                                   \
      ::std::fprintf(stderr, "EXPECT failed %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
      ::std::exit(1);                                                                   \
    }                                                                                   \
  } while (0)

#define CUDA_OK(call) EXPECT((call) == cudaSuccess)

using namespace cuda::experimental::sharded;

namespace
{

// One trial per thread-0 launch: sample n of [0, len) (population = iota, so
// sampled values ARE indices) with the given strategy and seed.
__global__ void sample_trial_kernel(
  reserved::__probe_sampler which,
  const int* pop,
  ::std::size_t len,
  ::std::size_t n,
  unsigned long long seed,
  int* out,
  ::std::size_t* out_count,
  ::std::size_t* idx_scratch)
{
  if (threadIdx.x != 0 || blockIdx.x != 0)
  {
    return;
  }
  ::cuda::std::philox4x64 gen{seed};
  auto* end = reserved::__probe_sample_n(
    which, pop, pop + len, out, n, gen, ::cuda::std::span<::std::size_t>{idx_scratch, n});
  *out_count = static_cast<::std::size_t>(end - out);
}

struct trial_buffers
{
  int* d_pop;
  int* d_out;
  ::std::size_t* d_count;
  ::std::size_t* d_scratch;
  ::std::size_t len, n;

  trial_buffers(::std::size_t len_, ::std::size_t n_)
      : len(len_)
      , n(n_)
  {
    ::std::vector<int> iota(len);
    for (::std::size_t i = 0; i < len; i++)
    {
      iota[i] = static_cast<int>(i);
    }
    CUDA_OK(cudaMalloc(&d_pop, len * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_out, (n + 1) * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_count, sizeof(::std::size_t)));
    CUDA_OK(cudaMalloc(&d_scratch, (n + 1) * sizeof(::std::size_t)));
    CUDA_OK(cudaMemcpy(d_pop, iota.data(), len * sizeof(int), cudaMemcpyHostToDevice));
  }
  ~trial_buffers()
  {
    cudaFree(d_pop);
    cudaFree(d_out);
    cudaFree(d_count);
    cudaFree(d_scratch);
  }

  // Run one trial; returns the drawn indices (values == indices).
  ::std::vector<int> run(reserved::__probe_sampler which, unsigned long long seed) const
  {
    sample_trial_kernel<<<1, 1>>>(which, d_pop, len, n, seed, d_out, d_count, d_scratch);
    CUDA_OK(cudaDeviceSynchronize());
    ::std::size_t count = 0;
    CUDA_OK(cudaMemcpy(&count, d_count, sizeof(count), cudaMemcpyDeviceToHost));
    ::std::vector<int> got(count);
    CUDA_OK(cudaMemcpy(got.data(), d_out, count * sizeof(int), cudaMemcpyDeviceToHost));
    return got;
  }
};

void check_shape(const ::std::vector<int>& got, ::std::size_t expected_count, ::std::size_t len)
{
  EXPECT(got.size() == expected_count);
  for (::std::size_t i = 0; i < got.size(); i++)
  {
    EXPECT(got[i] >= 0 && static_cast<::std::size_t>(got[i]) < len);
    if (i > 0)
    {
      EXPECT(got[i] > got[i - 1]); // population order, distinct
    }
  }
}

void test_shapes_and_clamp()
{
  const ::std::size_t len = 1000, n = 37;
  trial_buffers t(len, n);
  for (auto which : {reserved::__probe_sampler::__floyd, reserved::__probe_sampler::__systematic})
  {
    check_shape(t.run(which, 42), n, len);
  }
  // n >= len: whole population, both strategies.
  trial_buffers small(16, 64);
  for (auto which : {reserved::__probe_sampler::__floyd, reserved::__probe_sampler::__systematic})
  {
    const auto got = small.run(which, 7);
    EXPECT(got.size() == 16);
    for (int i = 0; i < 16; i++)
    {
      EXPECT(got[static_cast<::std::size_t>(i)] == i);
    }
  }
}

// The documented statistical distinction, demonstrated: len % n != 0 leaves
// a tail systematic can never draw; Floyd reaches everything at near-uniform
// frequency.
void test_tail_reachability()
{
  const ::std::size_t len = 1003, n = 40; // stride 25, tail = indices >= 1000
  const ::std::size_t trials = 400;
  trial_buffers t(len, n);

  ::std::vector<int> floyd_hits(len, 0), sys_hits(len, 0);
  for (::std::size_t s = 0; s < trials; s++)
  {
    for (int v : t.run(reserved::__probe_sampler::__floyd, 1000 + s))
    {
      floyd_hits[static_cast<::std::size_t>(v)]++;
    }
    for (int v : t.run(reserved::__probe_sampler::__systematic, 1000 + s))
    {
      sys_hits[static_cast<::std::size_t>(v)]++;
    }
  }
  // Systematic: the tail (indices >= n * stride = 1000) is unreachable.
  EXPECT(sys_hits[1000] == 0 && sys_hits[1001] == 0 && sys_hits[1002] == 0);
  // Floyd: the same tail is reachable, and per-index frequency is near the
  // uniform expectation n/len (~0.04 * 400 = 16 hits; loose 4x bounds).
  EXPECT(floyd_hits[1000] > 0 && floyd_hits[1001] > 0 && floyd_hits[1002] > 0);
  long long total = 0;
  int max_hits = 0, min_hits = 1 << 30;
  for (::std::size_t i = 0; i < len; i++)
  {
    total += floyd_hits[i];
    max_hits = floyd_hits[i] > max_hits ? floyd_hits[i] : max_hits;
    min_hits = floyd_hits[i] < min_hits ? floyd_hits[i] : min_hits;
  }
  EXPECT(total == static_cast<long long>(trials * n));
  const double expect_per = static_cast<double>(trials * n) / static_cast<double>(len);
  EXPECT(min_hits > expect_per / 4.0);
  EXPECT(max_hits < expect_per * 4.0);
}

void test_default_selection()
{
  unsetenv("CUDAX_SHARDED_PROBE_SAMPLER");
  EXPECT(reserved::__default_probe_sampler() == reserved::__probe_sampler::__floyd);
  setenv("CUDAX_SHARDED_PROBE_SAMPLER", "systematic", 1);
  EXPECT(reserved::__default_probe_sampler() == reserved::__probe_sampler::__systematic);
  setenv("CUDAX_SHARDED_PROBE_SAMPLER", "floyd", 1);
  EXPECT(reserved::__default_probe_sampler() == reserved::__probe_sampler::__floyd);
  unsetenv("CUDAX_SHARDED_PROBE_SAMPLER");
}

} // namespace

int main()
{
  CUDA_OK(cudaSetDevice(0));
  test_shapes_and_clamp();
  test_tail_reachability();
  test_default_selection();
  ::std::printf("probe_sampling: all tests passed\n");
  return 0;
}
