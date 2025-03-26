//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Algorithm construct, to embed a CUDA graph within a task
 */

#pragma once

#include <cuda/experimental/__stf/allocators/pooled_allocator.cuh>
#include <cuda/experimental/__stf/internal/context.cuh>

#include <map>

namespace cuda::experimental::stf
{

/**
 * @brief Algorithms are a mechanism to implement reusable task sequences implemented by the means of CUDA graphs nested
 * within a task.
 *
 * The underlying CUDA graphs are cached so that they are temptatively reused
 * when the algorithm is run again. Nested algorithms are internally implemented as child graphs.
 */
class algorithm
{
private:
  template <typename context_t, typename... Deps>
  class runner_impl
  {
  public:
    runner_impl(context_t& _ctx, algorithm& _alg, task_dep<Deps>... _deps)
        : alg(_alg)
        , ctx(_ctx)
        , deps(::std::make_tuple(mv(_deps)...)) {};

    template <typename Fun>
    void operator->*(Fun&& fun)
    {
      // We cannot use ::std::apply with a lambda function here instead
      // because this would use extended lambda functions within a lambda
      // function which is prohibited
      call_with_tuple_impl(::std::forward<Fun>(fun), ::std::index_sequence_for<Deps...>{});
    }

  private:
    // Helper function to call fun with context and unpacked tuple arguments
    template <typename Fun, ::std::size_t... Idx>
    void call_with_tuple_impl(Fun&& fun, ::std::index_sequence<Idx...>)
    {
      // We may simply execute the algorithm within the existing context
      // if we do not want to generate sub-graphs (eg. to analyze the
      // advantage of using such algorithms)
      if (getenv("CUDASTF_ALGORITHM_INLINE"))
      {
        alg.run_inline(::std::forward<Fun>(fun), ctx, ::std::get<Idx>(deps)...);
      }
      else
      {
        alg.run_as_task(::std::forward<Fun>(fun), ctx, ::std::get<Idx>(deps)...);
      }
    }

    algorithm& alg;
    context_t& ctx;
    ::std::tuple<task_dep<Deps>...> deps;
  };

public:
  algorithm(::std::string _symbol = "algorithm")
      : symbol(mv(_symbol))
  {}

  /* Inject the execution of the algorithm within a CUDA graph */
  template <typename Fun, typename parent_ctx_t, typename... Args>
  void run_in_graph(Fun fun, parent_ctx_t& parent_ctx, cudaGraph_t graph, Args... args)
  {
    auto argsTuple = ::std::make_tuple(args...);
    ::cuda::experimental::stf::hash<decltype(argsTuple)> hasher;
    size_t hashValue = hasher(argsTuple);

    ::std::shared_ptr<cudaGraph_t> inner_graph;

    if (auto search = graph_cache.find(hashValue); search != graph_cache.end())
    {
      inner_graph = search->second;
    }
    else
    {
      graph_ctx gctx(parent_ctx.async_resources());

      // Useful for tools
      gctx.set_parent_ctx(parent_ctx);
      gctx.get_dot()->set_ctx_symbol("algo: " + symbol);

      auto current_place = gctx.default_exec_place();

      // Transform an instance into a new logical data
      auto logify = [&gctx, &current_place](auto x) {
        // Our infrastructure currently does not like to work with
        // constant types for the data interface so we pretend this is
        // a modifiable data if necessary
        return gctx.logical_data(to_rw_type_of(x), current_place.affine_data_place());
      };

      // Transform the tuple of instances into a tuple of logical data
      auto logicalArgsTuple = ::std::apply(
        [&](auto&&... args) {
          return ::std::tuple(logify(::std::forward<decltype(args)>(args))...);
        },
        argsTuple);

      // call fun(gctx, ...logical data...)
      ::std::apply(fun, ::std::tuple_cat(::std::make_tuple(gctx), logicalArgsTuple));

      inner_graph = gctx.finalize_as_graph();

      // TODO validate that the graph is reusable before storing it !
      // fprintf(stderr, "CACHE graph...\n");
      graph_cache[hashValue] = inner_graph;
    }

    cudaGraphNode_t c;
    cuda_safe_call(cudaGraphAddChildGraphNode(&c, graph, nullptr, 0, *inner_graph));
  }

  /* This simply executes the algorithm within the existing context. This
   * makes it possible to observe the impact of an algorithm by disabling it in
   * practice (without bloating the code with both the algorithm and the original
   * code) */
  template <typename context_t, typename Fun, typename... Deps>
  void run_inline(Fun fun, context_t& ctx, task_dep<Deps>... deps)
  {
    ::std::apply(fun,
                 ::std::tuple_cat(::std::make_tuple(ctx), ::std::make_tuple(logical_data<Deps>(deps.get_data())...)));
  }

  /* Helper to run the algorithm in a stream_ctx */
  template <typename Fun, typename... Deps>
  void run_as_task(Fun fun, stream_ctx& ctx, task_dep<Deps>... deps)
  {
    ctx.task(deps...).set_symbol(symbol)->*[this, &fun, &ctx](cudaStream_t stream, Deps... args) {
      this->run(fun, ctx, stream, args...);
    };
  }

  /* Helper to run the algorithm in a graph_ctx */
  template <typename Fun, typename... Deps>
  void run_as_task(Fun fun, graph_ctx& ctx, task_dep<Deps>... deps)
  {
    ctx.task(deps...).set_symbol(symbol)->*[this, &fun, &ctx](cudaGraph_t g, Deps... args) {
      this->run_in_graph(fun, ctx, g, args...);
    };
  }

  /**
   * @brief Executes `fun` within a task that takes a pack of dependencies
   *
   * As an alternative, the run_as_task_dynamic may take a variable number of dependencies
   */
  template <typename Fun, typename... Deps>
  void run_as_task(Fun fun, context& ctx, task_dep<Deps>... deps)
  {
    ::std::visit(
      [&](auto& actual_ctx) {
        this->run_as_task(fun, actual_ctx, deps...);
      },
      ctx.payload);
  }

  /* Helper to run the algorithm in a stream_ctx */
  template <typename Fun>
  void run_as_task_dynamic(Fun fun, stream_ctx& ctx, const ::std::vector<task_dep_untyped>& deps)
  {
    auto t = ctx.task();
    for (auto& d : deps)
    {
      t.add_deps(d);
    }

    t.set_symbol(symbol);

    t->*[this, &fun, &ctx, &t](cudaStream_t stream) {
      this->run_dynamic(fun, ctx, stream, t);
    };
  }

  /* Helper to run the algorithm in a graph_ctx */
  template <typename Fun>
  void run_as_task_dynamic(Fun /* fun */, graph_ctx& /* ctx */, const ::std::vector<task_dep_untyped>& /* deps */)
  {
    /// TODO
    abort();
  }

  /**
   * @brief Executes `fun` within a task that takes a vector of untyped dependencies.
   *
   * This is an alternative for run_as_task which may take a variable number of dependencies
   */
  template <typename Fun>
  void run_as_task_dynamic(Fun fun, context& ctx, const ::std::vector<task_dep_untyped>& deps)
  {
    ::std::visit(
      [&](auto& actual_ctx) {
        this->run_as_task_dynamic(fun, actual_ctx, deps);
      },
      ctx.payload);
  }

  /**
   * @brief Helper to use algorithm using the ->* idiom instead of passing the implementation as an argument of
   * run_as_task
   *
   * example:
   *    algorithm alg;
   *    alg.runner(ctx, lX.read(), lY.rw())->*[](context inner_ctx, logical_data<slice<double>> X,
   * logical_data<slice<double>> Y) { inner_ctx.parallel_for(Y.shape(), X.rw(), Y.rw())->*[]__device__(size_t i, auto
   * x, auto y) { y(i) = 2.0*x(i);
   *        };
   *    };
   *
   *
   * Which is equivalent to:
   * auto fn = [](context inner_ctx, logical_data<slice<double>> X,  logical_data<slice<double>> Y) {
   *     inner_ctx.parallel_for(Y.shape(), X.rw(), Y.rw())->*[]__device__(size_t i, auto x, auto y) {
   *         y(i) = 2.0*x(i);
   *     }
   * };
   *
   * algorithm alg;
   * alg.run_as_task(fn, ctx, lX.read(), lY.rw());
   */
  template <typename context_t, typename... Deps>
  runner_impl<context_t, Deps...> runner(context_t& ctx, task_dep<Deps>... deps)
  {
    return runner_impl(ctx, *this, mv(deps)...);
  }

  /* Execute the algorithm as a CUDA graph and launch this graph in a CUDA
   * stream */
  template <typename Fun, typename parent_ctx_t, typename... Args>
  void run(Fun fun, parent_ctx_t& parent_ctx, cudaStream_t stream, Args... args)
  {
    auto argsTuple = ::std::make_tuple(args...);
    graph_ctx gctx(parent_ctx.async_resources());

    // Useful for tools
    gctx.set_parent_ctx(parent_ctx);
    gctx.get_dot()->set_ctx_symbol("algo: " + symbol);

    // This creates an adapter which "redirects" allocations to the CUDA stream API
    auto wrapper = stream_adapter(gctx, stream);

    gctx.update_uncached_allocator(wrapper.allocator());

    auto current_place = gctx.default_exec_place();

    // Transform an instance into a new logical data
    auto logify = [&gctx, &current_place](auto x) {
      // Our infrastructure currently does not like to work with constant
      // types for the data interface so we pretend this is a modifiable
      // data if necessary
      return gctx.logical_data(to_rw_type_of(x), current_place.affine_data_place());
    };

    // Transform the tuple of instances into a tuple of logical data
    auto logicalArgsTuple = ::std::apply(
      [&](auto&&... args) {
        return ::std::tuple(logify(::std::forward<decltype(args)>(args))...);
      },
      argsTuple);

    // call fun(gctx, ...logical data...)
    ::std::apply(fun, ::std::tuple_cat(::std::make_tuple(gctx), logicalArgsTuple));

    ::std::shared_ptr<cudaGraph_t> gctx_graph = gctx.finalize_as_graph();

    // Try to reuse existing exec graphs...
    ::std::shared_ptr<cudaGraphExec_t> eg = nullptr;
    bool found                            = false;
    for (::std::shared_ptr<cudaGraphExec_t>& pe : cached_exec_graphs[stream])
    {
      found = reserved::try_updating_executable_graph(*pe, *gctx_graph);
      if (found)
      {
        eg = pe;
        break;
      }
    }

    if (!found)
    {
      auto cudaGraphExecDeleter = [](cudaGraphExec_t* pGraphExec) {
        cudaGraphExecDestroy(*pGraphExec);
      };
      ::std::shared_ptr<cudaGraphExec_t> res(new cudaGraphExec_t, cudaGraphExecDeleter);

      dump_algorithm(gctx_graph);

      cuda_try(cudaGraphInstantiateWithFlags(res.get(), *gctx_graph, 0));

      eg = res;

      cached_exec_graphs[stream].push_back(eg);
    }

    cuda_safe_call(cudaGraphLaunch(*eg, stream));

    // Free resources allocated through the adapter
    wrapper.clear();
  }

  /* Contrary to `run`, we here have a dynamic set of dependencies for the
   * task, so fun does not take a pack of data instances as a parameter */
  template <typename Fun, typename parent_ctx_t, typename task_t>
  void run_dynamic(Fun fun, parent_ctx_t& parent_ctx, cudaStream_t stream, task_t& t)
  {
    graph_ctx gctx(parent_ctx.async_resources());

    // Useful for tools
    gctx.set_parent_ctx(parent_ctx);
    gctx.get_dot()->set_ctx_symbol("algo: " + symbol);

    gctx.set_allocator(block_allocator<pooled_allocator>(gctx));

    auto current_place = gctx.default_exec_place();

    fun(gctx, t);

    ::std::shared_ptr<cudaGraph_t> gctx_graph = gctx.finalize_as_graph();

    // Try to reuse existing exec graphs...
    ::std::shared_ptr<cudaGraphExec_t> eg = nullptr;
    bool found                            = false;
    for (::std::shared_ptr<cudaGraphExec_t>& pe : cached_exec_graphs[stream])
    {
      found = reserved::try_updating_executable_graph(*pe, *gctx_graph);
      if (found)
      {
        eg = pe;
        break;
      }
    }

    if (!found)
    {
      auto cudaGraphExecDeleter = [](cudaGraphExec_t* pGraphExec) {
        cudaGraphExecDestroy(*pGraphExec);
      };
      ::std::shared_ptr<cudaGraphExec_t> res(new cudaGraphExec_t, cudaGraphExecDeleter);

      dump_algorithm(gctx_graph);

      cuda_try(cudaGraphInstantiateWithFlags(res.get(), *gctx_graph, 0));

      eg = res;

      cached_exec_graphs[stream].push_back(eg);
    }

    cuda_safe_call(cudaGraphLaunch(*eg, stream));
  }

private:
  // Generate a DOT output of a CUDA graph using CUDA
  void dump_algorithm(const ::std::shared_ptr<cudaGraph_t>& gctx_graph)
  {
    if (getenv("CUDASTF_DUMP_ALGORITHMS"))
    {
      static int print_to_dot_cnt = 0; // Warning: not thread-safe
      ::std::string filename      = "algo_" + symbol + "_" + ::std::to_string(print_to_dot_cnt++) + ".dot";
      cudaGraphDebugDotPrint(*gctx_graph, filename.c_str(), cudaGraphDebugDotFlags(0));
    }
  }

  ::std::map<cudaStream_t, ::std::vector<::std::shared_ptr<cudaGraphExec_t>>> cached_exec_graphs;

  // Cache executable graphs
  ::std::unordered_map<size_t, ::std::shared_ptr<cudaGraphExec_t>> exec_graph_cache;

  // Cache CUDA graphs
  ::std::unordered_map<size_t, ::std::shared_ptr<cudaGraph_t>> graph_cache;

  ::std::string symbol;
};

} // end namespace cuda::experimental::stf
