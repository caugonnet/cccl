//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Minimal test for stf_ctx_create_ex() with has_stream=1 (caller-provided
// CUDA stream) on the stream backend, with NO async_resources handle shared
// across contexts. The goal is to verify the chaining contract:
//
//   "The stream passed to context creation asynchronously depends on
//    everything the tasks did (transitively); consecutive contexts on
//    the same caller stream therefore serialize with each other without
//    any explicit sync."
//
// We submit a single token-backed task that runs a "slow" kernel writing
// value V into a device buffer. We do this in two back-to-back contexts
// that share ONLY the caller stream (no handle). If the contract holds,
// the final buffer must contain the LAST write (value 2).

#include <cuda_runtime.h>

#include <vector>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

namespace
{

// Writes `value` into every slot of `arr`. The inner busy loop widens the
// kernel window so that a failure to chain ctx2-after-ctx1 is observable:
// ctx1 is still running when ctx2's kernel races in.
__global__ void slow_set_kernel(int* arr, int n, int value, int iters)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
  {
    return;
  }
  int acc = 0;
  // Busy loop to keep the kernel resident on the SM for a while.
  for (int i = 0; i < iters; ++i)
  {
    acc += (i * 1103515245 + 12345) & 0x7fffffff;
  }
  // Side-effect: ensure the compiler does not elide the loop, then commit
  // the intended `value` (independent of acc) to the buffer.
  arr[tid] = value + (acc & 0);
}

void submit_set(stf_ctx_handle ctx, int* d_arr, int n, int value, int iters)
{
  stf_logical_data_handle tok = stf_token(ctx);
  REQUIRE(tok != nullptr);
  stf_logical_data_set_symbol(tok, "tok");

  stf_task_handle t = stf_task_create(ctx);
  REQUIRE(t != nullptr);
  stf_task_set_symbol(t, "slow_set");
  stf_task_add_dep(t, tok, STF_RW);
  stf_task_start(t);

  CUstream s = stf_task_get_custream(t);
  REQUIRE(s != nullptr);

  const int threads = 128;
  const int blocks  = (n + threads - 1) / threads;
  slow_set_kernel<<<blocks, threads, 0, (cudaStream_t) s>>>(d_arr, n, value, iters);

  stf_task_end(t);
  stf_task_destroy(t);

  stf_logical_data_destroy(tok);
}

} // namespace

namespace
{

// Submits `K` concurrent token-tasks in a single context; each writes
// `value` into its own slice of `d_arr`. Mirrors the failing MLP shape:
// multiple independent tokens per context, so STF spreads the kernels
// across several pool streams and the context is not sequentially
// self-draining.
void run_ctx_k_concurrent(cudaStream_t s, int* d_arr, int N, int K, int value, int iters)
{
  stf_ctx_options opts{};
  opts.backend    = STF_BACKEND_STREAM;
  opts.has_stream = 1;
  opts.stream     = s;
  opts.handle     = nullptr; // <-- key: NO shared async_resources_handle

  stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
  REQUIRE(ctx != nullptr);

  const int per = N / K;
  for (int k = 0; k < K; ++k)
  {
    stf_logical_data_handle tok = stf_token(ctx);
    REQUIRE(tok != nullptr);

    stf_task_handle t = stf_task_create(ctx);
    REQUIRE(t != nullptr);
    stf_task_add_dep(t, tok, STF_RW);
    stf_task_start(t);

    CUstream ts         = stf_task_get_custream(t);
    const int threads   = 128;
    const int blocks    = (per + threads - 1) / threads;
    int* slice          = d_arr + k * per;
    slow_set_kernel<<<blocks, threads, 0, (cudaStream_t) ts>>>(slice, per, value, iters);

    stf_task_end(t);
    stf_task_destroy(t);
    stf_logical_data_destroy(tok);
  }

  stf_ctx_finalize(ctx);
}

} // namespace

C2H_TEST("stf_ctx_create_ex: 1 token per context, back-to-back, stream-only", "[context][stream]")
{
  constexpr int N     = 1 << 14;
  constexpr int ITERS = 1 << 18;

  cudaStream_t s{};
  REQUIRE(cudaStreamCreate(&s) == cudaSuccess);

  int* d_arr = nullptr;
  REQUIRE(cudaMalloc(&d_arr, N * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(d_arr, 0, N * sizeof(int), s) == cudaSuccess);

  for (int iter = 0; iter < 20; ++iter)
  {
    {
      stf_ctx_options opts{};
      opts.backend    = STF_BACKEND_STREAM;
      opts.has_stream = 1;
      opts.stream     = s;
      opts.handle     = nullptr;

      stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
      REQUIRE(ctx != nullptr);
      submit_set(ctx, d_arr, N, /*value=*/1, ITERS);
      stf_ctx_finalize(ctx);
    }
    {
      stf_ctx_options opts{};
      opts.backend    = STF_BACKEND_STREAM;
      opts.has_stream = 1;
      opts.stream     = s;
      opts.handle     = nullptr;

      stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
      REQUIRE(ctx != nullptr);
      submit_set(ctx, d_arr, N, /*value=*/2, ITERS);
      stf_ctx_finalize(ctx);
    }

    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    int h_arr[16]{};
    REQUIRE(cudaMemcpy(h_arr, d_arr, sizeof(h_arr), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < (int) (sizeof(h_arr) / sizeof(int)); ++i)
    {
      INFO("iter=" << iter << " i=" << i << " value=" << h_arr[i]);
      REQUIRE(h_arr[i] == 2);
    }
  }

  REQUIRE(cudaFree(d_arr) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
}

namespace
{

// More faithful MLP mimic: K concurrent tokens, each with T chained tasks
// (sequential RW on the same token), so each token effectively owns a
// chain of T slow kernels on one pool stream. With K tokens, we get K
// parallel chains of length T. The key behavior we're probing: when the
// context (no shared handle) destructs while some of these chains are
// still in flight, are the pool streams / pool-owned events / pool-owned
// mempool freed too early?
void run_ctx_k_chains(
  cudaStream_t s, int* d_arr, int N, int K, int chain_len, int value, int iters)
{
  stf_ctx_options opts{};
  opts.backend    = STF_BACKEND_STREAM;
  opts.has_stream = 1;
  opts.stream     = s;
  opts.handle     = nullptr; // fresh internal async_resources_handle owned by ctx

  stf_ctx_handle ctx = stf_ctx_create_ex(&opts);
  REQUIRE(ctx != nullptr);

  const int per = N / K;
  std::vector<stf_logical_data_handle> toks(K);
  for (int k = 0; k < K; ++k)
  {
    toks[k] = stf_token(ctx);
    REQUIRE(toks[k] != nullptr);
  }

  for (int step = 0; step < chain_len; ++step)
  {
    for (int k = 0; k < K; ++k)
    {
      stf_task_handle t = stf_task_create(ctx);
      REQUIRE(t != nullptr);
      stf_task_add_dep(t, toks[k], STF_RW);
      stf_task_start(t);

      CUstream ts       = stf_task_get_custream(t);
      const int threads = 128;
      const int blocks  = (per + threads - 1) / threads;
      int* slice        = d_arr + k * per;
      slow_set_kernel<<<blocks, threads, 0, (cudaStream_t) ts>>>(slice, per, value, iters);

      stf_task_end(t);
      stf_task_destroy(t);
    }
  }

  for (int k = 0; k < K; ++k)
  {
    stf_logical_data_destroy(toks[k]);
  }

  // Non-blocking finalize: records outbound events on `s` and releases
  // the context. If the ctx-owned async_resources_handle destructor runs
  // before in-flight pool work drains, that's the bug we expect to see.
  stf_ctx_finalize(ctx);
}

} // namespace

C2H_TEST(
  "stf_ctx_create_ex: K chains of T tasks per token, back-to-back, stream-only, no handle "
  "(mirrors the MLP-ensemble shape: pool lifetime vs in-flight outbound events)",
  "[context][stream][tokens][lifetime]")
{
  constexpr int N         = 1 << 16;
  constexpr int K         = 8;   // concurrent chains
  constexpr int CHAIN_LEN = 20;  // 4 steps * 5 kernels per step in the MLP
  constexpr int ITERS     = 1 << 18;

  cudaStream_t s{};
  REQUIRE(cudaStreamCreate(&s) == cudaSuccess);

  int* d_arr = nullptr;
  REQUIRE(cudaMalloc(&d_arr, N * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(d_arr, 0, N * sizeof(int), s) == cudaSuccess);

  for (int iter = 0; iter < 20; ++iter)
  {
    run_ctx_k_chains(s, d_arr, N, K, CHAIN_LEN, /*value=*/1, ITERS);
    run_ctx_k_chains(s, d_arr, N, K, CHAIN_LEN, /*value=*/2, ITERS);

    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    std::vector<int> h_arr(N, 0);
    REQUIRE(cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

    int mismatches  = 0;
    int first_bad_i = -1;
    int first_bad_v = 0;
    for (int i = 0; i < N; ++i)
    {
      if (h_arr[i] != 2)
      {
        ++mismatches;
        if (first_bad_i < 0)
        {
          first_bad_i = i;
          first_bad_v = h_arr[i];
        }
      }
    }
    INFO("iter=" << iter
                 << " mismatches=" << mismatches
                 << " first_bad_idx=" << first_bad_i
                 << " first_bad_val=" << first_bad_v);
    REQUIRE(mismatches == 0);
  }

  REQUIRE(cudaFree(d_arr) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
}

C2H_TEST(
  "stf_ctx_create_ex: K concurrent tokens per context, back-to-back, stream-only "
  "(mirrors the failing MLP-ensemble shape)",
  "[context][stream][tokens]")
{
  constexpr int N     = 1 << 16;
  constexpr int K     = 8;   // concurrent tokens per context
  constexpr int ITERS = 1 << 18;

  cudaStream_t s{};
  REQUIRE(cudaStreamCreate(&s) == cudaSuccess);

  int* d_arr = nullptr;
  REQUIRE(cudaMalloc(&d_arr, N * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(d_arr, 0, N * sizeof(int), s) == cudaSuccess);

  // Run multiple rounds. Each round: ctx1 writes 1, ctx2 writes 2, both
  // share the caller stream `s`, neither uses an async_resources_handle
  // -> each gets its own pool, so the two contexts' task streams are
  // disjoint. The inbound chaining (ctx2's first tasks must wait on
  // anything pending on `s`) is the only thing keeping this race-free.
  for (int iter = 0; iter < 20; ++iter)
  {
    run_ctx_k_concurrent(s, d_arr, N, K, /*value=*/1, ITERS);
    run_ctx_k_concurrent(s, d_arr, N, K, /*value=*/2, ITERS);

    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    std::vector<int> h_arr(N, 0);
    REQUIRE(cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

    int mismatches   = 0;
    int first_bad_i  = -1;
    int first_bad_v  = 0;
    for (int i = 0; i < N; ++i)
    {
      if (h_arr[i] != 2)
      {
        ++mismatches;
        if (first_bad_i < 0)
        {
          first_bad_i = i;
          first_bad_v = h_arr[i];
        }
      }
    }
    INFO("iter=" << iter
                 << " mismatches=" << mismatches
                 << " first_bad_idx=" << first_bad_i
                 << " first_bad_val=" << first_bad_v);
    REQUIRE(mismatches == 0);
  }

  REQUIRE(cudaFree(d_arr) == cudaSuccess);
  REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
}
