// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// keys.cu with a launch-shape axis: the sort of a sharded array is one SortKeys per locality
// domain, run concurrently; this benchmark measures (and tunes) that per-domain call in the shape
// it runs in. "W" reproduces keys.cu. See nvbench_helper/launch_shape.cuh.

#include <launch_shape.cuh>
#include <nvbench_helper.cuh>

// %//RANGE//% TUNE_RADIX_BITS bits 8:9:1
#define TUNE_RADIX_BITS 8

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

#include "policy_selector.h"

template <typename T, typename OffsetT>
void radix_sort_keys_shaped(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using value_t = cub::NullType;
  if constexpr (!fits_in_default_shared_memory<T, value_t, OffsetT, cub::SortOrder::Ascending>())
  {
    return;
  }

  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));
  shaped_lanes lanes{str_to_launch_shape(state.get_string("Shape")), state};
  if (lanes.skipped)
  {
    return;
  }

  thrust::device_vector<T> buffer_1 = generate(elements, entropy);
  thrust::device_vector<T> buffer_2(elements, thrust::no_init);
  lane_buffers<T> in  = make_lane_buffers(lanes, buffer_1);
  lane_buffers<T> out = make_lane_buffers(lanes, buffer_2);
  const auto per_lane = static_cast<OffsetT>(lanes.items(elements));

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  exec_shaped(state, lanes, [&](int lane, cudaStream_t stream, caching_allocator_t& alloc) {
    auto env = cub_bench_env_on(
      alloc,
      stream
#if !TUNE_BASE
      ,
      cuda::execution::tune(policy_selector<T, value_t, OffsetT>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceRadixSort::SortKeys, "SortKeys failed", in.ptr(lane), out.ptr(lane), per_lane, 0, sizeof(T) * 8, env);
  });
}

NVBENCH_BENCH_TYPES(radix_sort_keys_shaped, NVBENCH_TYPE_AXES(fundamental_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.201"})
  .add_string_axis("Shape", {"W", "2L", "2NL"});
