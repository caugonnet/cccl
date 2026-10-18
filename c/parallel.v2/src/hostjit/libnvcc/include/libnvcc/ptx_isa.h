//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// PTX ISA version libnvcc asks LLVM for (`-target-feature +ptxNN`) when
// compiling device code for a given SM. LLVM's NVPTX backend rejects an SM it
// considers newer than the requested ISA (e.g. "PTX version 8.5 does not
// support target 'sm_103'"), and the toolkit's ptxas must accept the ISA, so
// the value has to grow with the SM. Kept header-only so the table is unit
// tested without loading LLVM (test/freestanding/test_host_portability.cpp).
//
// Floor is 7.8: generated device code uses PTX 7.6 features (e.g. `bmsk`)
// that older ISAs cannot assemble even for sm_75 / sm_80.
// SM libnvcc hands to LLVM (`-target-cpu sm_NN`) for a given device SM. The
// NVPTX backend only knows the SMs listed in llvm/lib/Target/NVPTX/NVPTX.td
// (LLVM 22.1.1: ..., 90, 100, 101, 103, 110, 120, 121); any other SM makes
// `createTargetMachine` fail with a fatal error. An unknown SM is compiled for
// the base SM of its family (sm_107 -> sm_100, the largest known SM not above
// sm_X0): base PTX is forward compatible inside the family, and ptxas/nvJitLink
// still finalize the cubin for the real SM (see `-arch=sm_NN` in compiler.cpp),
// so the fatbin carries the device's own architecture.
constexpr int libnvcc_llvm_sm_for_sm(int sm_version)
{
  constexpr int known[] = {20, 21, 30, 32, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 103, 110, 120, 121};
  for (int sm : known)
  {
    if (sm == sm_version)
    {
      return sm_version;
    }
  }
  const int family_base = (sm_version / 10) * 10;
  int best = known[0];
  for (int sm : known)
  {
    if (sm <= family_base && sm > best)
    {
      best = sm;
    }
  }
  return best;
}

constexpr int libnvcc_ptx_isa_for_sm(int sm_version)
{
  sm_version = libnvcc_llvm_sm_for_sm(sm_version); // the ISA has to match what LLVM is asked to target
  if (sm_version == 103 || sm_version >= 121) // GB300 and the sm_121+ family: PTX 8.8 (CUDA 13)
  {
    return 88;
  }
  if (sm_version >= 120) // Blackwell consumer (sm_120/120a/120f): PTX 8.7
  {
    return 87;
  }
  if (sm_version >= 100) // Blackwell datacenter (sm_100/101): PTX 8.6 (LLVM 22's NVPTX minimum for sm_100)
  {
    return 86;
  }
  if (sm_version >= 90) // Hopper: PTX 8.0
  {
    return 80;
  }
  return 78;
}
