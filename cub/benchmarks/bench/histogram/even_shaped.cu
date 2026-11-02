// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// even.cu with a launch-shape axis: one HistogramEven per locality domain over its shard, run
// concurrently. The histogram sweep grid is sized from the SM count, so this is the flavor where
// the per-domain shape has its own optimum. See nvbench_helper/launch_shape.cuh.

#include <launch_shape.cuh>
#include <nvbench_helper.cuh>

#include "histogram_common.cuh"

// %RANGE% TUNE_ITEMS ipt 4:28:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_RLE_COMPRESS rle 0:1:1
// %RANGE% TUNE_WORK_STEALING ws 0:1:1
// %RANGE% TUNE_MEM_PREFERENCE mem 0:2:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_LOAD_ALGORITHM_ID laid 0:2:1
// %RANGE% TUNE_VEC_SIZE_POW vec 0:2:1

template <typename SampleT, typename CounterT, typename OffsetT>
static void even_shaped(nvbench::state& state, nvbench::type_list<SampleT, CounterT, OffsetT>)
{
  const auto entropy   = str_to_entropy(state.get_string("Entropy"));
  const auto elements  = state.get_int64("Elements{io}");
  const auto num_bins  = state.get_int64("Bins");
  const int num_levels = static_cast<int>(num_bins) + 1;
  if constexpr (cuda::std::is_integral_v<SampleT>)
  {
    if (num_bins > static_cast<int64_t>(cuda::std::numeric_limits<SampleT>::max()))
    {
      state.skip("Number of bins exceeds what LevelT (= SampleT) can represent");
      return;
    }
  }
  shaped_lanes lanes{str_to_launch_shape(state.get_string("Shape")), state};
  if (lanes.skipped)
  {
    return;
  }

  const SampleT lower_level            = 0;
  const SampleT upper_level            = get_upper_level<SampleT>(num_bins, elements);
  thrust::device_vector<SampleT> input = generate(elements, entropy, lower_level, upper_level);
  thrust::device_vector<CounterT> hist(static_cast<std::size_t>(num_bins) * lanes.lanes);
  lane_buffers<SampleT> in = make_lane_buffers(lanes, input);
  CounterT* d_hist         = thrust::raw_pointer_cast(hist.data());
  const auto per_lane      = static_cast<OffsetT>(lanes.items(static_cast<std::size_t>(elements)));

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements);
  state.add_global_memory_writes<CounterT>(num_bins);

  exec_shaped(state, lanes, [&](int lane, cudaStream_t stream, caching_allocator_t& alloc) {
    auto env = cub_bench_env_on(
      alloc,
      stream
#if !TUNE_BASE
      ,
      cuda::execution::tune(bench_policy_selector<SampleT, 1, 1>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceHistogram::HistogramEven,
      "HistogramEven failed",
      in.ptr(lane),
      d_hist + static_cast<std::size_t>(lane) * num_bins,
      num_levels,
      lower_level,
      upper_level,
      per_lane,
      env);
  });
}

using counter_types     = nvbench::type_list<int32_t>;
using some_offset_types = nvbench::type_list<int32_t>;

#ifdef TUNE_SampleT
using sample_types = nvbench::type_list<TUNE_SampleT>;
#else // !defined(TUNE_SampleT)
using sample_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double>;
#endif // TUNE_SampleT

NVBENCH_BENCH_TYPES(even_shaped, NVBENCH_TYPE_AXES(sample_types, counter_types, some_offset_types))
  .set_name("base")
  .set_type_axes_names({"SampleT{ct}", "CounterT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_axis("Bins", {32, 128, 2048, 2097152})
  .add_string_axis("Entropy", {"0.201", "1.000"})
  .add_string_axis("Shape", {"W", "2L", "2NL"});
