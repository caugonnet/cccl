# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Example showing how to use generate to fill an output array with values
produced by a nullary generator.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Prepare the output array.
d_out = cp.empty(5, dtype=np.int32)


# Define a nullary generator that produces a constant value.
def constant_42():
    return 42


# Fill the output array.
cuda.compute.generate(d_out, constant_42, len(d_out))

# Verify the result.
result = d_out.get()
expected = np.full(5, 42, dtype=np.int32)

np.testing.assert_array_equal(result, expected)
print(f"Generate result: {result}")
