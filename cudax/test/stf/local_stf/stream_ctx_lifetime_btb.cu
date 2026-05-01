//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Probe the lifetime of an async_resources_handle that is implicitly
 *        owned by a `stream_ctx(stream)` (i.e. no user-provided handle).
 *
 * Reproduces the shape of the failing Python MLP-ensemble test
 * (`probe_warp_cache_ablation.py`, case `NOhandle, no sync`):
 *
 *   - Two back-to-back `stream_ctx(stream)` invocations on the *same* caller
 *     stream, with *no* shared async_resources_handle and *no* explicit
 *     stream synchronization between them.
 *   - Each context submits K concurrent token chains of T chained RW tasks
 *     (so STF spreads the work across several pool streams), each task
 *     running a long, busy-loop kernel.
 *
 * If the ctx-owned async_resources_handle (and the pool streams / pool
 * mempool / pool event pool it owns) is destroyed synchronously at
 * `ctx.finalize()` while kernels are still in-flight on the pool streams,
 * we expect either:
 *   - a correctness failure on the final buffer (second context's writes
 *     get overwritten / reordered), or
 *   - a CUDA error (observable under compute-sanitizer).
 *
 * The expected "fix" behavior is either:
 *   - explicit `cudaStreamSynchronize(stream)` between the two calls, or
 *   - a shared `async_resources_handle` that outlives both contexts.
 */

#include <cuda/experimental/stf.cuh>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

using namespace cuda::experimental::stf;

namespace
{
// Writes `value` into every slot of `slice`. The clock64()-based busy
// wait widens the kernel window so the kernel stays resident on the SM
// long enough for ctx.finalize() to return *before* the kernel completes.
__global__ void slow_set_kernel(int* slice, int n, int value, long long ns)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
  {
    return;
  }
  long long start = clock64();
  while (clock64() - start < ns)
  {
    // busy wait
  }
  slice[tid] = value;
}

// Submits, inside a fresh `stream_ctx(stream)` (no handle), K concurrent
// token chains of length `chain_len`. Each chain is a linear RW chain on
// one token, so STF binds it to a single pool stream and the K chains
// execute concurrently.
void run_ctx_k_chains(cudaStream_t s, int* d_arr, int N, int K, int chain_len, int value, long long ns)
{
  // *** Key shape: fresh stream_ctx(stream) bound to the caller's stream,
  // with an *implicit* async_resources_handle owned by this context. ***
  stream_ctx ctx(s);

  std::vector<logical_data<void_interface>> toks;
  toks.reserve(K);
  for (int k = 0; k < K; ++k)
  {
    toks.push_back(ctx.token());
  }

  const int per = N / K;

  for (int step = 0; step < chain_len; ++step)
  {
    for (int k = 0; k < K; ++k)
    {
      int* slice = d_arr + k * per;
      ctx.task(toks[k].rw())->*[=](cudaStream_t ts) {
        const int threads = 128;
        const int blocks  = (per + threads - 1) / threads;
        slow_set_kernel<<<blocks, threads, 0, ts>>>(slice, per, value, ns);
      };
    }
  }

  // Non-blocking finalize: records outbound events back onto `s` and
  // tears down the context. With no user handle, the ctx-owned
  // async_resources_handle (and its pool streams) get released here.
  ctx.finalize();
}

int run_once(int iter, int N, int K, int chain_len, long long ns, bool sync_between, bool with_handle)
{
  cudaStream_t s{};
  if (cudaStreamCreate(&s) != cudaSuccess)
  {
    std::fprintf(stderr, "cudaStreamCreate failed\n");
    return 1;
  }

  int* d_arr = nullptr;
  if (cudaMalloc(&d_arr, N * sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "cudaMalloc failed\n");
    return 1;
  }
  cudaMemsetAsync(d_arr, 0, N * sizeof(int), s);

  auto submit_pair = [&](auto&& call) {
    call(1);
    if (sync_between)
    {
      cudaStreamSynchronize(s);
    }
    call(2);
  };

  if (with_handle)
  {
    async_resources_handle h;
    auto call = [&, ns](int value) {
      stream_ctx ctx(s, h);
      std::vector<logical_data<void_interface>> toks;
      toks.reserve(K);
      for (int k = 0; k < K; ++k)
      {
        toks.push_back(ctx.token());
      }
      const int per = N / K;
      for (int step = 0; step < chain_len; ++step)
      {
        for (int k = 0; k < K; ++k)
        {
          int* slice = d_arr + k * per;
          ctx.task(toks[k].rw())->*[=](cudaStream_t ts) {
            const int threads = 128;
            const int blocks  = (per + threads - 1) / threads;
            slow_set_kernel<<<blocks, threads, 0, ts>>>(slice, per, value, ns);
          };
        }
      }
      ctx.finalize();
    };
    submit_pair(call);
  }
  else
  {
    auto call = [&, ns](int value) {
      run_ctx_k_chains(s, d_arr, N, K, chain_len, value, ns);
    };
    submit_pair(call);
  }

  cudaStreamSynchronize(s);

  std::vector<int> h_arr(N, 0);
  cudaMemcpy(h_arr.data(), d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

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

  std::printf("  iter=%d mismatches=%d first_bad_idx=%d first_bad_val=%d\n", iter, mismatches, first_bad_i, first_bad_v);

  cudaFree(d_arr);
  cudaStreamDestroy(s);
  return mismatches;
}

void run_case(
  const char* label,
  int iters_outer,
  int N,
  int K,
  int chain_len,
  long long ns,
  bool sync_between,
  bool with_handle,
  int& total_mismatches)
{
  std::printf("== %s ==\n", label);
  for (int it = 0; it < iters_outer; ++it)
  {
    total_mismatches += run_once(it, N, K, chain_len, ns, sync_between, with_handle);
  }
}
} // namespace

int main()
{
  // Shape roughly mirrors the MLP ensemble: E=K=8 concurrent chains,
  // 4 steps * 5 kernels = 20 sequential kernels per chain. Each kernel
  // uses a big busy-loop so the context destructor races with in-flight
  // pool work when there is no sync/handle.
  constexpr int N         = 1 << 16;
  constexpr int K         = 16;
  constexpr int CHAIN_LEN = 40;
  // ~5ms per kernel at 1GHz clock rate -> ~200ms of in-flight work per chain
  // when we return from ctx.finalize().
  constexpr long long BUSY_CYCLES = 5'000'000;
  constexpr int OUTER             = 20;

  int fail_nohandle_nosync = 0;
  int fail_nohandle_sync   = 0;
  int fail_handle_nosync   = 0;

  run_case(
    "NOhandle, no sync  (expected buggy)",
    OUTER,
    N,
    K,
    CHAIN_LEN,
    BUSY_CYCLES,
    /*sync_between=*/false,
    /*with_handle=*/false,
    fail_nohandle_nosync);
  run_case(
    "NOhandle, sync between",
    OUTER,
    N,
    K,
    CHAIN_LEN,
    BUSY_CYCLES,
    /*sync_between=*/true,
    /*with_handle=*/false,
    fail_nohandle_sync);
  run_case(
    "handle,   no sync",
    OUTER,
    N,
    K,
    CHAIN_LEN,
    BUSY_CYCLES,
    /*sync_between=*/false,
    /*with_handle=*/true,
    fail_handle_nosync);

  std::printf("\nSummary (total mismatched slots across %d iters each):\n", OUTER);
  std::printf("  NOhandle, no sync  : %d\n", fail_nohandle_nosync);
  std::printf("  NOhandle, sync     : %d\n", fail_nohandle_sync);
  std::printf("  handle,   no sync  : %d\n", fail_handle_nosync);

  // Intentionally don't assert here: we want to *observe* the bug. The
  // CI-friendly variants (sync / handle) must stay at 0.
  if (fail_nohandle_sync != 0 || fail_handle_nosync != 0)
  {
    return 1;
  }
  return 0;
}
