# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Generate example demonstrating the object API.
"""

import cupy as cp
import numpy as np

import cuda.compute

# Prepare the output array.
dtype = np.int32
d_output = cp.empty(4, dtype=dtype)


# Define a nullary generator.
def constant_7():
    return 7


# Create the generate object (compiles the kernel once).
generator = cuda.compute.make_generate(d_output, constant_7)

# Perform the generation (can be called repeatedly without recompilation).
generator(d_output, constant_7, len(d_output))

# Verify the result.
expected_result = np.array([7, 7, 7, 7], dtype=dtype)
actual_result = d_output.get()
np.testing.assert_array_equal(actual_result, expected_result)
print("Generate object example completed successfully")
