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
 * @brief Computes Eigenvector Centrality for vertices within a graph
 *
 */

#include <cuda/experimental/stf.cuh>

#include <cmath>
#include <vector>

using namespace cuda::experimental::stf;

/**
 * @brief Gathers the (A + I) power-iteration contribution for a single vertex: the sum of the
 *        previous-iteration centrality of every in-neighbor, plus the vertex's own previous
 *        centrality (the "+ I" identity term).
 *
 * @param idx        The index of the vertex for which the contribution is being calculated.
 * @param loffsets   Slice containing the offset vector of the CSR representation.
 * @param lnonzeros  Slice containing the non-zero elements (neighbors) vector of the CSR representation.
 * @param lold       Slice containing the previous-iteration centrality values.
 * @return           The updated (un-normalized) centrality value for vertex idx.
 */
__device__ float gather_and_add_identity(
  int idx, const slice<const int>& loffsets, const slice<const int>& lnonzeros, const slice<const float>& lold)
{
  float sum = 0.0f;
  for (int i = loffsets[idx]; i < loffsets[idx + 1]; i++)
  {
    sum += lold[lnonzeros[i]];
  }
  return sum + lold[idx];
}

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving example: while_graph_scope is only available since CUDA 12.4.\n");
  return 0;
#else
  stackable_ctx ctx;

  // A small deterministic graph with heterogeneous in-degree (built with a simple LCG so it is
  // reproducible without an external RNG dependency). A regular ring/cycle graph is deliberately
  // avoided here: its tiny spectral gap makes plain power iteration stall for many iterations
  // without a clean convergence story.
  const int num_vertices = 96;
  std::vector<int> offsets(num_vertices + 1, 0);
  std::vector<int> nonzeros;
  // Note: the degree and neighbor draws below use the LCG's high bits, not
  // its low bits. A power-of-two-modulus LCG has a very short period in its
  // low-order bits (state % 8 here would be nearly constant), which would
  // silently produce an almost-regular graph.
  unsigned state = 0x2545F491u;
  for (int v = 0; v < num_vertices; v++)
  {
    state         = state * 1664525u + 1013904223u;
    int in_degree = 3 + static_cast<int>((state >> 24) % 8); // in-degree between 3 and 10
    for (int k = 0; k < in_degree; k++)
    {
      state = state * 1664525u + 1013904223u;
      nonzeros.push_back(static_cast<int>((state >> 16) % num_vertices));
    }
    offsets[v + 1] = static_cast<int>(nonzeros.size());
  }

  float init_centrality = 1.0f / num_vertices;
  float tolerance       = 1e-6f;
  int NITER             = 100;

  // output centralities for each vertex
  std::vector<float> centrality(num_vertices, init_centrality);
  std::vector<float> old_centrality(num_vertices);

  auto loffsets    = ctx.logical_data(&offsets[0], offsets.size());
  auto lnonzeros   = ctx.logical_data(&nonzeros[0], nonzeros.size());
  auto lcentrality = ctx.logical_data(&centrality[0], centrality.size());
  auto lold        = ctx.logical_data(&old_centrality[0], old_centrality.size());
  auto lnorm_sq    = ctx.logical_data(shape_of<scalar_view<float>>());
  auto ldiff       = ctx.logical_data(shape_of<scalar_view<float>>());
  auto liter       = ctx.logical_data(shape_of<scalar_view<int>>());

  // Initialize iteration counter
  ctx.parallel_for(box(1), liter.write())->*[] __device__(size_t, auto iter) {
    *iter = 0;
  };

  {
    auto while_guard = ctx.while_graph_scope();

    // Save the current centrality as "old" before overwriting it.
    ctx.parallel_for(lcentrality.shape(), lold.write(), lcentrality.read())
        ->*[] __device__(size_t i, auto old_centrality, auto centrality) {
              old_centrality(i) = centrality(i);
            };

    // Apply (A + I) and accumulate the squared L2 norm of the new (un-normalized) centrality.
    ctx.parallel_for(
      box(num_vertices),
      loffsets.read(),
      lnonzeros.read(),
      lold.read(),
      lcentrality.write(),
      lnorm_sq.reduce(reducer::sum<float>{}))
        ->*[] __device__(size_t idx, auto loffsets, auto lnonzeros, auto lold, auto lcentrality, auto& norm_sq) {
              float updated    = gather_and_add_identity(static_cast<int>(idx), loffsets, lnonzeros, lold);
              lcentrality(idx) = updated;
              norm_sq += updated * updated;
            };

    // Normalize by the L2 norm and accumulate the L1 convergence difference.
    ctx.parallel_for(
      lcentrality.shape(), lcentrality.rw(), lold.read(), lnorm_sq.read(), ldiff.reduce(reducer::sum<float>{}))
        ->*[] __device__(size_t i, auto centrality, auto old_centrality, auto norm_sq, auto& diff) {
              centrality(i) /= sqrtf(*norm_sq);
              diff += fabsf(centrality(i) - old_centrality(i));
            };

    while_guard.update_cond(ldiff.read(), liter.rw())->*[NITER, tolerance] __device__(auto diff, auto iter) {
      bool converged   = (*diff < tolerance);
      bool max_reached = ((*iter)++ >= NITER); // Maximum iteration limit
      return !converged && !max_reached; // Continue if not converged and under limit
    };
  }

  ctx.finalize();

  /*      CHECKING FOR ANSWER CORRECTNESS      */
  bool all_finite_and_nonnegative = true;
  for (int64_t i = 0; i < num_vertices; i++)
  {
    if (!(centrality[i] >= 0.0f) || !std::isfinite(centrality[i]))
    {
      all_finite_and_nonnegative = false;
    }
  }
  EXPECT(all_finite_and_nonnegative);

  // Independent double-precision host reference of the same (A + I) power iteration, used only
  // to sanity-check the device result rather than to reproduce its exact rounding.
  std::vector<double> ref(num_vertices, 1.0 / num_vertices);
  std::vector<double> ref_old(num_vertices);
  int ref_iterations = 0;
  for (; ref_iterations < NITER; ref_iterations++)
  {
    ref_old = ref;
    for (int v = 0; v < num_vertices; v++)
    {
      double sum = 0.0;
      for (int e = offsets[v]; e < offsets[v + 1]; e++)
      {
        sum += ref_old[nonzeros[e]];
      }
      ref[v] = sum + ref_old[v];
    }
    double norm_sq = 0.0;
    for (int v = 0; v < num_vertices; v++)
    {
      norm_sq += ref[v] * ref[v];
    }
    double norm = std::sqrt(norm_sq);
    double diff = 0.0;
    for (int v = 0; v < num_vertices; v++)
    {
      ref[v] /= norm;
      diff += std::abs(ref[v] - ref_old[v]);
    }
    if (diff < tolerance)
    {
      ref_iterations++;
      break;
    }
  }

  bool matches_reference = true;
  for (int64_t i = 0; i < num_vertices; i++)
  {
    double delta = std::abs(static_cast<double>(centrality[i]) - ref[i]);
    if (delta > 1e-3 * std::abs(ref[i]) + 1e-4)
    {
      matches_reference = false;
    }
  }
  EXPECT(matches_reference);

  printf("Eigenvector centrality answer is %s (converged, host reference matches within tolerance).\n",
         matches_reference ? "correct" : "not correct");

  printf("Eigenvector Centrality Results:\n");
  for (size_t i = 0; i < centrality.size(); ++i)
  {
    printf("Vertex %zu: %f\n", i, centrality[i]);
  }

  return 0;
#endif // !_CCCL_CTK_BELOW(12, 4)
}
