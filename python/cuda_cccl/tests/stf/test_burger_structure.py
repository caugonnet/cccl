# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Structural test for the Burger solver nesting pattern (4 levels)
using trivial math (no real physics, no sparse matrices).

Nesting:
  for outer in range(outer_iters):       # Python host loop
    graph_scope:                          # level 1
      repeat(substeps):                   # level 2
        while_loop (outer_cond):          # level 3 -- "Newton"
          X += 0.1
          while_loop (inner_cond):        # level 4 -- "CG"
            Y += 0.05
            inner_cond = |Y - inner_target| > tol
          outer_cond = |X - outer_target| > tol

This verifies that graph_scope > repeat > while > while works before
adding real Burger physics on top.

Requires CUDA 12.4+ (conditional graph nodes).
"""

import numpy as np
import torch
from pytorch_task import pytorch_task

import cuda.stf as stf


def test_4level_nesting():
    """
    4-level: for > graph_scope > repeat > while("Newton") > while("CG").

    Outer while ("Newton"): X walks toward outer_target in steps of 0.1.
    Inner while ("CG"): Y walks toward inner_target in steps of 0.05.
    After inner converges, Y is reset (simulating CG producing a fresh delta).

    repeat runs the outer-while 3 times; the outer target advances each
    repeat iteration so X keeps climbing.

    Expected final X ~ 3.0 (3 repeat iters, each converging to target 1, 2, 3).
    """
    n = 64
    X_host = np.zeros(n, dtype=np.float64)
    Y_host = np.zeros(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")
    lY = ctx.logical_data(Y_host, name="Y")

    outer_target_host = np.array([1.0], dtype=np.float64)
    inner_target_host = np.array([1.0], dtype=np.float64)
    louter_target = ctx.logical_data(outer_target_host, name="outer_target")
    linner_target = ctx.logical_data(inner_target_host, name="inner_target")
    linner_target.set_read_only()

    louter_res = ctx.logical_data_empty((1,), np.float64, name="outer_res")
    linner_res = ctx.logical_data_empty((1,), np.float64, name="inner_res")

    outer_step = 0.1
    inner_step = 0.05
    tol = 0.05

    substeps = 3
    outer_iters = 2

    for _outer in range(outer_iters):
        with ctx.graph_scope():                              # level 1
            with ctx.repeat(substeps):                       # level 2
                with ctx.while_loop() as newton_loop:        # level 3
                    # --- "Newton step": advance X ---
                    with pytorch_task(ctx, lX.rw()) as (tX,):
                        tX[:] += outer_step

                    # --- "CG solve": advance Y toward inner target ---
                    with pytorch_task(ctx, lY.rw()) as (tY,):
                        tY.zero_()

                    with ctx.while_loop() as cg_loop:        # level 4
                        with pytorch_task(ctx, lY.rw()) as (tY,):
                            tY[:] += inner_step

                        with pytorch_task(
                            ctx,
                            lY.read(), linner_target.read(), linner_res.write()
                        ) as (tY, tTgt, tRes):
                            tRes[0] = torch.max(torch.abs(tY - tTgt[0]))

                        cg_loop.continue_while(linner_res, ">", tol)

                    # --- outer condition ---
                    with pytorch_task(
                        ctx,
                        lX.read(), louter_target.read(), louter_res.write()
                    ) as (tX, tTgt, tRes):
                        tRes[0] = torch.max(torch.abs(tX - tTgt[0]))

                    newton_loop.continue_while(louter_res, ">", tol)

                # After each "Newton" convergence, bump target
                with pytorch_task(ctx, louter_target.rw()) as (tTgt,):
                    tTgt[0] += 1.0

    ctx.finalize()

    total_repeats = outer_iters * substeps
    expected_X = float(total_repeats)
    print(f"X[0] = {X_host[0]:.4f}  (expected ~{expected_X})")
    print(f"Y[0] = {Y_host[0]:.4f}  (expected ~1.0)")
    assert np.allclose(X_host, expected_X, atol=outer_step + tol), \
        f"Expected ~{expected_X}, got {X_host[0]}"
    print("4-level nesting test PASSED")


def test_while_inside_while_minimal():
    """
    Minimal while-inside-while (no repeat, no graph_scope around it).

    graph_scope:
      while_loop (outer):
        X += 0.5
        while_loop (inner):
          Y += 0.1
          inner_cond = |Y - 1.0| > tol
        outer_cond = |X - 2.0| > tol

    Expected: X ~ 2.0, Y ~ 1.0 (inner converges to 1 on each outer iter,
    but Y is reset each outer iter).
    """
    n = 32
    X_host = np.zeros(n, dtype=np.float64)
    Y_host = np.zeros(n, dtype=np.float64)

    ctx = stf.stackable_context()
    lX = ctx.logical_data(X_host, name="X")
    lY = ctx.logical_data(Y_host, name="Y")

    louter_res = ctx.logical_data_empty((1,), np.float64, name="outer_res")
    linner_res = ctx.logical_data_empty((1,), np.float64, name="inner_res")

    tol = 0.05

    with ctx.graph_scope():
        with ctx.while_loop() as outer_loop:
            with pytorch_task(ctx, lX.rw()) as (tX,):
                tX[:] += 0.5

            # Reset Y before inner loop
            with pytorch_task(ctx, lY.rw()) as (tY,):
                tY.zero_()

            with ctx.while_loop() as inner_loop:
                with pytorch_task(ctx, lY.rw()) as (tY,):
                    tY[:] += 0.1

                with pytorch_task(ctx, lY.read(), linner_res.write()) as (tY, tRes):
                    tRes[0] = torch.max(torch.abs(tY - 1.0))

                inner_loop.continue_while(linner_res, ">", tol)

            with pytorch_task(ctx, lX.read(), louter_res.write()) as (tX, tRes):
                tRes[0] = torch.max(torch.abs(tX - 2.0))

            outer_loop.continue_while(louter_res, ">", tol)

    ctx.finalize()

    print(f"X[0] = {X_host[0]:.4f}  (expected ~2.0)")
    print(f"Y[0] = {Y_host[0]:.4f}  (expected ~1.0)")
    assert np.allclose(X_host, 2.0, atol=0.5 + tol), f"Expected ~2.0, got {X_host[0]}"
    assert np.allclose(Y_host, 1.0, atol=0.1 + tol), f"Expected ~1.0, got {Y_host[0]}"
    print("while-inside-while minimal test PASSED")


if __name__ == "__main__":
    print("=== test_while_inside_while_minimal ===")
    test_while_inside_while_minimal()

    print("\n=== test_4level_nesting ===")
    test_4level_nesting()

    print("\nAll structure tests passed!")
