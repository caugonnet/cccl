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
 * @brief Reproducer for the abort observed in the Python binding when a
 *        stackable token is used inside a push()/pop_prologue(_shared)()
 *        scope:
 *
 *            ctx = stf.stackable_context()
 *            tok = ctx.token()
 *            ctx.push()
 *            with ctx.task(tok.write()): ...
 *            with ctx.task(tok.read()):  ...
 *            g = ctx.pop_prologue_shared()
 *
 *        aborts inside STF with:
 *            Data interface type mismatch.
 *            Assumed: cuda::experimental::stf::void_interface
 *            Actual:  mdspan<char, extents<long unsigned int, ...>,
 *                           layout_stride>
 *
 *        This translates that exact sequence to the native cudax::stf C++
 *        API so the bug is reproducible in pure C++.
 *
 *        The existing stackable_token.cu test is adjacent but does NOT hit
 *        the bug because it explicitly calls ltoken.push(access_mode::rw)
 *        right after sctx.push() -- i.e. it imports the token into the
 *        nested scope before any task references it. The Python binding
 *        does not do that: tok.write() / tok.read() are the first touches.
 *        Removing the ltoken.push() call in that test should reproduce the
 *        same failure.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// Same shape as stackable_token.cu but WITHOUT the explicit
// ltoken.push(access_mode::rw) that primes the token in the nested scope.
// This is what the Python binding does -- tok.write()/tok.read() are the
// first references to the token inside the pushed graph scope.
static void repro_push_pop_token_no_explicit_push()
{
  stackable_ctx sctx;

  auto ltoken = sctx.token();

  sctx.push();
  // NOTE: deliberately NOT calling ltoken.push(access_mode::rw); that's
  // the difference vs. the existing stackable_token.cu regression test.
  sctx.task(ltoken.rw())->*[](cudaStream_t, auto) {
    // token-only task: no payload to touch.
  };
  sctx.task(ltoken.read())->*[](cudaStream_t, auto) {};
  sctx.pop();

  sctx.finalize();
}

// Same minimal pattern as above, but using write()/read() separately (the
// exact methods the Python binding uses).
static void repro_push_pop_token_write_then_read()
{
  stackable_ctx sctx;

  auto ltoken = sctx.token();

  sctx.push();
  sctx.task(ltoken.write())->*[](cudaStream_t, auto) {};
  sctx.task(ltoken.read())->*[](cudaStream_t, auto) {};
  sctx.pop();

  sctx.finalize();
}

// Minimal C++ analog of the simplest C-facade failing shape:
// push -> single task(tok.write()) -> pop.
static void repro_push_single_write_token()
{
  stackable_ctx sctx;

  auto ltoken = sctx.token();

  sctx.push();
  sctx.task(ltoken.write())->*[](cudaStream_t, auto) {};
  sctx.pop();

  sctx.finalize();
}

// Non-shared pop_prologue flavor: ctx.push() -> task(tok.write()) ->
// task(tok.read()) -> pop_prologue() (+ pop_epilogue). In the C facade this
// aborts identically to the shared flavor, so this variant pins down
// whether the C++ side behaves differently depending on shared-ness.
static void repro_push_pop_prologue_token()
{
  stackable_ctx sctx;

  auto ltoken = sctx.token();

  sctx.push();
  sctx.task(ltoken.write())->*[](cudaStream_t, auto) {};
  sctx.task(ltoken.read())->*[](cudaStream_t, auto) {};

  auto h = sctx.pop_prologue();
  for (int k = 0; k < 3; ++k)
  {
    h.launch();
  }
  sctx.pop_epilogue();

  sctx.finalize();
}

// Full-fidelity mirror of run_stf_unified in the Python mockup:
// ctx.push() -> task(tok.write()) -> task(tok.read()) -> pop_prologue_shared().
static void repro_push_pop_prologue_shared_token()
{
  stackable_ctx sctx;

  auto ltoken = sctx.token();

  sctx.push();
  sctx.task(ltoken.write())->*[](cudaStream_t, auto) {};
  sctx.task(ltoken.read())->*[](cudaStream_t, auto) {};

  // Expected failure site: the prologue instantiation tries to materialise
  // the token logical_data and hits the void_interface vs
  // mdspan<char,...,layout_stride> type mismatch.
  auto g = sctx.pop_prologue_shared();

  // We don't expect to reach here, but if we ever do, validate the handle
  // and do a couple of relaunches to make sure the graph is well-formed.
  for (int k = 0; k < 3; ++k)
  {
    g.launch();
  }
  // Release the shared graph before finalize() so pop_epilogue() runs.
  g.reset();

  sctx.finalize();
}

// Sanity: the same push/pop_prologue_shared shape works when we use a real
// logical_data instead of a token. This matches the run_stf_unified_ld
// workaround in the Python mockup.
static void workaround_push_pop_prologue_shared_logical_data()
{
  stackable_ctx sctx;

  auto lA = sctx.logical_data(shape_of<slice<int>>(8));

  // Produce an initial value so the data is defined before the pushed scope
  // reads it.
  sctx.parallel_for(lA.shape(), lA.write())->*[] __device__(size_t i, auto a) {
    a(i) = 0;
  };

  sctx.push();
  sctx.task(lA.rw())->*[](cudaStream_t, auto) {};
  sctx.task(lA.read())->*[](cudaStream_t, auto) {};
  auto g = sctx.pop_prologue_shared();

  for (int k = 0; k < 3; ++k)
  {
    g.launch();
  }
  g.reset();

  sctx.finalize();
}

int main(int argc, char** argv)
{
  int which = argc > 1 ? atoi(argv[1]) : 4;
  switch (which)
  {
    case 0:
      fprintf(stderr, "[variant 0] push -> task(tok.rw()) -> task(tok.read()) -> pop\n");
      repro_push_pop_token_no_explicit_push();
      break;
    case 1:
      fprintf(stderr, "[variant 1] push -> task(tok.write()) -> task(tok.read()) -> pop\n");
      repro_push_pop_token_write_then_read();
      break;
    case 2:
      fprintf(stderr, "[variant 2] push -> task(tok.write()) -> pop  [minimal C-facade shape]\n");
      repro_push_single_write_token();
      break;
    case 3:
      fprintf(stderr, "[variant 3] push -> task(tok.write()) -> task(tok.read()) -> pop_prologue (non-shared)\n");
      repro_push_pop_prologue_token();
      break;
    case 4:
      fprintf(stderr, "[variant 4] push -> task(tok.write()) -> task(tok.read()) -> pop_prologue_shared\n");
      repro_push_pop_prologue_shared_token();
      break;
    case 5:
      fprintf(stderr, "[variant 5] (workaround) push -> task(lA.rw()) -> task(lA.read()) -> pop_prologue_shared\n");
      workaround_push_pop_prologue_shared_logical_data();
      break;
    default:
      fprintf(stderr, "unknown variant %d (valid: 0..5)\n", which);
      return 2;
  }
  fprintf(stderr, "[variant %d] OK\n", which);
  return 0;
}
