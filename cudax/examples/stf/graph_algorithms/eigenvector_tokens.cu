//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Eigenvector centrality where CUDASTF orchestrates external code and
 *        external memory: every dependency is a data-less token, every kernel
 *        is a plain __global__ function or a CUB call over caller-owned
 *        buffers and workspaces, the convergence decision stays on the
 *        device, and the solver graph is built once and replayed.
 *
 * Unlike the other graph examples, STF neither allocates the algorithm state
 * nor generates any kernel here. This is the integration shape for an
 * existing CUDA library: buffers and CUB temporary storage are allocated by
 * the caller before graph construction, tasks launch the library's kernels on
 * the stream they are given, tokens express the dependencies between those
 * launches, and update_cond consumes a device scalar produced by CUB so the
 * (A + I) power iteration runs entirely under a while_graph_scope with no
 * host round trip. The launchable graph is instantiated once with exec() and
 * can be replayed with launch() at will.
 */

#include <cub/device/device_reduce.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/experimental/stf.cuh>

#include <cmath>
#include <vector>

using namespace cuda::experimental::stf;

// The kernels below stand in for an existing library's optimized routines:
// they are ordinary __global__ functions that know nothing about CUDASTF.

__global__ void initialize_kernel(float* current, int* iteration, int num_vertices, float initial)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_vertices)
  {
    current[idx] = initial;
  }
  if (idx == 0)
  {
    *iteration = 0;
  }
}

__global__ void copy_kernel(float* dst, const float* src, int num_vertices)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_vertices)
  {
    dst[idx] = src[idx];
  }
}

// Pull-based neighbor sum: current[v] = sum of old[u] over in-neighbors u.
__global__ void
gather_kernel(const int* offsets, const int* nonzeros, const float* old, float* current, int num_vertices)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_vertices)
  {
    float sum = 0.0f;
    for (int e = offsets[idx]; e < offsets[idx + 1]; e++)
    {
      sum += old[nonzeros[e]];
    }
    current[idx] = sum;
  }
}

// The identity term of the (A + I) power iteration.
__global__ void add_kernel(float* current, const float* old, int num_vertices)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_vertices)
  {
    current[idx] += old[idx];
  }
}

__global__ void normalize_kernel(float* current, const float* norm_sq, int num_vertices)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_vertices)
  {
    current[idx] /= sqrtf(*norm_sq);
  }
}

// Transform-reduce inputs for CUB.
struct square_op
{
  const float* values;

  __device__ float operator()(int i) const
  {
    return values[i] * values[i];
  }
};

struct absolute_difference_op
{
  const float* current;
  const float* old;

  __device__ float operator()(int i) const
  {
    return fabsf(current[i] - old[i]);
  }
};

template <typename Op>
auto make_reduce_input(Op op)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator(0), op);
}

// The condition functor carries the raw pointers itself; the token
// dependencies only order it after the difference reduction. It is variadic
// because token instances are filtered out of the arguments.
struct continue_op
{
  const float* difference;
  int* iteration;
  float threshold;
  int max_iterations;

  template <typename... Args>
  __device__ bool operator()(Args...) const
  {
    ++(*iteration);
    return (*difference >= threshold) && (*iteration < max_iterations);
  }
};

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving example: while_graph_scope is only available since CUDA 12.4.\n");
  return 0;
#else
  // A deterministic expander-like graph with heterogeneous in-degrees
  // (4 to 11 scattered edges per vertex), stored as pull CSR. The large
  // spectral gap keeps power iteration converging at a useful rate, and the
  // degree spread makes the eigenvector non-trivial.
  const int num_vertices = 512;
  std::vector<int> offsets(num_vertices + 1, 0);
  std::vector<int> nonzeros;
  for (int v = 0; v < num_vertices; v++)
  {
    unsigned state = static_cast<unsigned>(v) * 2654435761u + 12345u;
    for (int k = 0; k < 4 + (v % 8); k++)
    {
      state = state * 1664525u + 1013904223u;
      nonzeros.push_back(static_cast<int>((state >> 8) % num_vertices));
    }
    offsets[v + 1] = static_cast<int>(nonzeros.size());
  }

  const float initial      = 1.0f / num_vertices;
  const float threshold    = 1e-6f; // absolute L1 threshold on the normalized iterate
  const int max_iterations = 100;
  const int block_size     = 256;
  const int grid_size      = (num_vertices + block_size - 1) / block_size;

  // The caller owns every device allocation, including the CUB workspaces.
  int *d_offsets, *d_nonzeros, *d_iteration;
  float *d_current, *d_old, *d_norm_sq, *d_difference;
  cuda_safe_call(cudaMalloc(&d_offsets, offsets.size() * sizeof(int)));
  cuda_safe_call(cudaMalloc(&d_nonzeros, nonzeros.size() * sizeof(int)));
  cuda_safe_call(cudaMalloc(&d_current, num_vertices * sizeof(float)));
  cuda_safe_call(cudaMalloc(&d_old, num_vertices * sizeof(float)));
  cuda_safe_call(cudaMalloc(&d_norm_sq, sizeof(float)));
  cuda_safe_call(cudaMalloc(&d_difference, sizeof(float)));
  cuda_safe_call(cudaMalloc(&d_iteration, sizeof(int)));
  cuda_safe_call(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
  cuda_safe_call(cudaMemcpy(d_nonzeros, nonzeros.data(), nonzeros.size() * sizeof(int), cudaMemcpyHostToDevice));

  size_t norm_bytes = 0;
  cuda_safe_call(
    cub::DeviceReduce::Sum(nullptr, norm_bytes, make_reduce_input(square_op{d_current}), d_norm_sq, num_vertices));
  size_t difference_bytes = 0;
  cuda_safe_call(cub::DeviceReduce::Sum(
    nullptr, difference_bytes, make_reduce_input(absolute_difference_op{d_current, d_old}), d_difference, num_vertices));
  void *d_norm_temp, *d_difference_temp;
  cuda_safe_call(cudaMalloc(&d_norm_temp, norm_bytes));
  cuda_safe_call(cudaMalloc(&d_difference_temp, difference_bytes));

  stackable_ctx ctx;

  // One token per externally-owned buffer: tokens carry the dependencies,
  // the closures carry the pointers.
  auto t_current    = ctx.token();
  auto t_old        = ctx.token();
  auto t_norm       = ctx.token();
  auto t_difference = ctx.token();
  auto t_iteration  = ctx.token();
  t_current.set_symbol("current");
  t_old.set_symbol("old");
  t_norm.set_symbol("norm_sq");
  t_difference.set_symbol("difference");
  t_iteration.set_symbol("iteration");

  // The solver scope is a block so its RAII pop (which releases the frozen
  // token imports) runs before ctx.finalize().
  {
    stackable_ctx::launchable_graph_scope solver{ctx};
    t_current.push(access_mode::rw);
    t_old.push(access_mode::rw);
    t_norm.push(access_mode::rw);
    t_difference.push(access_mode::rw);
    t_iteration.push(access_mode::rw);

    ctx.task(t_current.write(), t_iteration.write()).set_symbol("initialize")->*[&](cudaStream_t stream, auto...) {
      initialize_kernel<<<grid_size, block_size, 0, stream>>>(d_current, d_iteration, num_vertices, initial);
    };

    {
      auto loop = ctx.while_graph_scope();

      ctx.task(t_old.write(), t_current.read()).set_symbol("save old")->*[&](cudaStream_t stream, auto...) {
        copy_kernel<<<grid_size, block_size, 0, stream>>>(d_old, d_current, num_vertices);
      };
      ctx.task(t_old.read(), t_current.write()).set_symbol("gather")->*[&](cudaStream_t stream, auto...) {
        gather_kernel<<<grid_size, block_size, 0, stream>>>(d_offsets, d_nonzeros, d_old, d_current, num_vertices);
      };
      ctx.task(t_current.rw(), t_old.read()).set_symbol("add identity")->*[&](cudaStream_t stream, auto...) {
        add_kernel<<<grid_size, block_size, 0, stream>>>(d_current, d_old, num_vertices);
      };
      ctx.task(t_current.read(), t_norm.write()).set_symbol("norm (cub)")->*[&](cudaStream_t stream, auto...) {
        size_t bytes = norm_bytes;
        cuda_safe_call(cub::DeviceReduce::Sum(
          d_norm_temp, bytes, make_reduce_input(square_op{d_current}), d_norm_sq, num_vertices, stream));
      };
      ctx.task(t_current.rw(), t_norm.read()).set_symbol("normalize")->*[&](cudaStream_t stream, auto...) {
        normalize_kernel<<<grid_size, block_size, 0, stream>>>(d_current, d_norm_sq, num_vertices);
      };
      ctx.task(t_current.read(), t_old.read(), t_difference.write()).set_symbol("difference (cub)")
          ->*[&](cudaStream_t stream, auto...) {
                size_t bytes = difference_bytes;
                cuda_safe_call(cub::DeviceReduce::Sum(
                  d_difference_temp,
                  bytes,
                  make_reduce_input(absolute_difference_op{d_current, d_old}),
                  d_difference,
                  num_vertices,
                  stream));
              };

      loop.update_cond(t_difference.read(), t_iteration.rw())
          ->*continue_op{d_difference, d_iteration, threshold, max_iterations};
    }

    // Instantiate once, then replay: the initialization task is part of the
    // graph, so every launch() re-runs the full solve over the same buffers.
    solver.exec();
    solver.launch();
    solver.launch();
    cuda_safe_call(cudaDeviceSynchronize());
  }

  std::vector<float> centralities(num_vertices);
  int iterations = 0;
  cuda_safe_call(cudaMemcpy(centralities.data(), d_current, num_vertices * sizeof(float), cudaMemcpyDeviceToHost));
  cuda_safe_call(cudaMemcpy(&iterations, d_iteration, sizeof(int), cudaMemcpyDeviceToHost));

  ctx.finalize();

  // Host reference in double precision: the same (A + I) power iteration.
  std::vector<double> reference(num_vertices, 1.0 / num_vertices);
  std::vector<double> reference_old(num_vertices);
  for (int iter = 0; iter < max_iterations; iter++)
  {
    reference_old = reference;
    for (int v = 0; v < num_vertices; v++)
    {
      double sum = 0.0;
      for (int e = offsets[v]; e < offsets[v + 1]; e++)
      {
        sum += reference_old[nonzeros[e]];
      }
      reference[v] = sum + reference_old[v];
    }
    double norm_sq = 0.0;
    for (int v = 0; v < num_vertices; v++)
    {
      norm_sq += reference[v] * reference[v];
    }
    const double norm = std::sqrt(norm_sq);
    double difference = 0.0;
    for (int v = 0; v < num_vertices; v++)
    {
      reference[v] /= norm;
      difference += std::abs(reference[v] - reference_old[v]);
    }
    if (difference < threshold)
    {
      break;
    }
  }

  EXPECT(iterations > 0);
  EXPECT(iterations < max_iterations);
  for (int v = 0; v < num_vertices; v++)
  {
    EXPECT(std::abs(centralities[v] - reference[v]) < 1e-3 * std::abs(reference[v]) + 1e-5);
  }
  printf("Eigenvector centrality converged in %d iterations.\n", iterations);

  cuda_safe_call(cudaFree(d_offsets));
  cuda_safe_call(cudaFree(d_nonzeros));
  cuda_safe_call(cudaFree(d_current));
  cuda_safe_call(cudaFree(d_old));
  cuda_safe_call(cudaFree(d_norm_sq));
  cuda_safe_call(cudaFree(d_difference));
  cuda_safe_call(cudaFree(d_iteration));
  cuda_safe_call(cudaFree(d_norm_temp));
  cuda_safe_call(cudaFree(d_difference_temp));

  return 0;
#endif // !_CCCL_CTK_BELOW(12, 4)
}
