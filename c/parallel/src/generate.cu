//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/util_device.cuh>

#include <cstring>
#include <format>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

#include <cccl/c/generate.h>
#include <cccl/c/types.h>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/types.h>

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

// -----------------------------------------------------------------------
// Kernel state: packs {output_ptr, [user_op_state]} into a flat buffer
// matching the generate_wrapper struct layout in the NVRTC code.
// -----------------------------------------------------------------------

struct generate_default
{
  void* output;
  void* user_op;
};

struct generate_kernel_state
{
  std::variant<generate_default, std::unique_ptr<char[]>> storage;

  void* get()
  {
    return std::visit(
      [](auto&& v) -> void* {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<generate_default, T>)
        {
          return &v;
        }
        else
        {
          return v.get();
        }
      },
      storage);
  }
};

static generate_kernel_state make_generate_kernel_state(cccl_op_t op, cccl_iterator_t d_out)
{
  const size_t ptr_size = sizeof(void*);

  size_t user_size  = (cccl_op_kind_t::CCCL_STATEFUL == op.type) ? op.size : 0;
  size_t user_align = (cccl_op_kind_t::CCCL_STATEFUL == op.type) ? op.alignment : 0;

  size_t user_op_offset = ptr_size;
  if (user_size)
  {
    size_t misalign = (user_op_offset & (user_align - 1));
    if (misalign)
    {
      user_op_offset += user_align - misalign;
    }
  }

  size_t total_size = user_op_offset + user_size;

  // The output pointer from the iterator (always a raw pointer for generate)
  void* output_ptr = d_out.state;

  generate_default local{};
  char* buf = reinterpret_cast<char*>(&local);

  bool use_heap = sizeof(generate_default) < total_size;
  if (use_heap)
  {
    buf = new char[total_size];
  }

  std::memcpy(buf, &output_ptr, ptr_size);
  if (cccl_op_kind_t::CCCL_STATEFUL == op.type)
  {
    std::memcpy(buf + user_op_offset, op.state, user_size);
  }

  if (use_heap)
  {
    return generate_kernel_state{std::unique_ptr<char[]>{buf}};
  }
  else
  {
    return generate_kernel_state{local};
  }
}

// -----------------------------------------------------------------------
// Kernel launch
// -----------------------------------------------------------------------

static cudaError_t
Invoke(cccl_iterator_t d_out, size_t num_items, cccl_op_t op, int /*cc*/, CUfunction static_kernel, CUstream stream)
{
  if (num_items == 0)
  {
    return cudaSuccess;
  }

  auto state = make_generate_kernel_state(op, d_out);

  void* args[] = {&num_items, state.get()};

  const unsigned int thread_count = 256;
  const size_t items_per_block    = 512;
  const size_t block_sz           = cuda::ceil_div(num_items, items_per_block);

  if (block_sz > std::numeric_limits<unsigned int>::max())
  {
    return cudaErrorInvalidValue;
  }
  const unsigned int block_count = static_cast<unsigned int>(block_sz);

  check(cuLaunchKernel(static_kernel, block_count, 1, 1, thread_count, 1, 1, 0, stream, args, 0));

  return CubDebug(cudaPeekAtLastError());
}

// -----------------------------------------------------------------------
// NVRTC code generation
// -----------------------------------------------------------------------

static std::string get_generate_user_op(cccl_op_t user_op)
{
  bool stateful = cccl_op_kind_t::CCCL_STATEFUL == user_op.type;

  // Op uses void* convention — compatible with the Numba JIT void-ptr wrapper output.
  constexpr std::string_view op_format =
    R"XXX(
#if {0}
#  define _STATEFUL_USER_OP
#endif

#define _USER_OP {1}

#if defined(_STATEFUL_USER_OP)
extern "C" __device__ void _USER_OP(void*, void*);
#else
extern "C" __device__ void _USER_OP(void*);
#endif

#if defined(_STATEFUL_USER_OP)
struct __align__({2}) user_op_t {{
  char data[{3}];
#else
struct user_op_t {{
#endif

  __device__ void operator()(void* out) {{
#if defined(_STATEFUL_USER_OP)
    _USER_OP(&data, out);
#else
    _USER_OP(out);
#endif
  }}
}};
)XXX";

  return std::format(
    op_format,
    stateful, // 0
    user_op.name, // 1
    user_op.alignment, // 2
    user_op.size // 3
  );
}

static std::string get_generate_kernel(cccl_op_t user_op, cccl_iterator_t d_out)
{
  return std::format(
    R"XXX(
#include <cuda/std/iterator>
#include <cub/agent/agent_for.cuh>
#include <cub/device/dispatch/kernels/kernel_for_each.cuh>
#include <cub/device/dispatch/tuning/tuning_for.cuh>

struct __align__({2}) output_storage_t {{
  char data[{3}];
}};

{0}

struct generate_wrapper
{{
  output_storage_t* output;
  user_op_t user_op;

  __device__ void operator()(unsigned long long idx)
  {{
    user_op(static_cast<void*>(&output[idx]));
  }}
}};

using device_for_policy_selector = cub::detail::for_each::policy_selector;
)XXX",
    get_generate_user_op(user_op), // 0
    0, // 1 (unused, placeholder)
    d_out.value_type.alignment, // 2
    d_out.value_type.size // 3
  );
}

// -----------------------------------------------------------------------
// Kernel name
// -----------------------------------------------------------------------

struct generate_wrapper;

static std::string get_generate_kernel_name()
{
  std::string offset_t;
  std::string wrapper_t;
  check(cccl_type_name_from_nvrtc<generate_wrapper>(&wrapper_t));
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  return std::format("cub::detail::for_each::static_kernel<device_for_policy_selector, {0}, {1}>", offset_t, wrapper_t);
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

CUresult cccl_device_generate_build_ex(
  cccl_device_generate_build_result_t* build_ptr,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  const char* name = "test";

  const int cc = cc_major * 10 + cc_minor;

  const std::string kernel_name = get_generate_kernel_name();
  const std::string kernel_src  = get_generate_kernel(op, d_out);

  const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

  std::vector<const char*> args = {
    arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-DCUB_DISABLE_CDP"};

  cccl::detail::extend_args_with_build_config(args, config);

  constexpr size_t num_lto_args   = 2;
  const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

  std::string lowered_name;

  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};
  appender.append_operation(op);

  nvrtc_link_result result =
    begin_linking_nvrtc_program(num_lto_args, lopts)
      ->add_program(nvrtc_translation_unit{kernel_src, name})
      ->add_expression({kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({kernel_name, lowered_name})
      ->link_program()
      ->add_link_list(linkable_list)
      ->finalize_program();

  cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  check(cuLibraryGetKernel(&build_ptr->static_kernel, build_ptr->library, lowered_name.c_str()));

  build_ptr->cc         = cc;
  build_ptr->cubin      = (void*) result.data.release();
  build_ptr->cubin_size = result.size;

  return CUDA_SUCCESS;
}
catch (...)
{
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_generate(
  cccl_device_generate_build_result_t build, cccl_iterator_t d_out, uint64_t num_items, cccl_op_t op, CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;

  try
  {
    pushed           = try_push_context();
    auto exec_status = Invoke(d_out, num_items, op, build.cc, (CUfunction) build.static_kernel, stream);
    error            = static_cast<CUresult>(exec_status);
  }
  catch (...)
  {
    error = CUDA_ERROR_UNKNOWN;
  }

  if (pushed)
  {
    CUcontext dummy;
    cuCtxPopCurrent(&dummy);
  }

  return error;
}

CUresult cccl_device_generate_build(
  cccl_device_generate_build_result_t* build,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_generate_build_ex(
    build, d_out, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, nullptr);
}

CUresult cccl_device_generate_cleanup(cccl_device_generate_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
  check(cuLibraryUnload(build_ptr->library));

  return CUDA_SUCCESS;
}
catch (...)
{
  return CUDA_ERROR_UNKNOWN;
}
