// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Launch shapes: a tuning axis for algorithms that run inside one locality domain of a device
// (a green context owning a subset of the SMs, with its own memory). Shapes:
//
//   W    whole device: the benchmark's own stream, one call over all elements (today's benchmarks)
//   2L   every locality domain concurrently, each on a stream bound to that domain, over its own
//        memory, elements / domains per domain
//   2NL  every domain concurrently over its slice of whole-device (interleaved) memory
//
// Why a gate: at the sizes where tuning matters a per-domain kernel lasts tens of microseconds, the
// same order as the launch skew between streams, so naively launched domains run partly in sequence
// and the measurement silently degrades to "one domain alone". Every domain stream first runs a
// one-block gate kernel that arrives on a counter and spins until (domains + 1) arrivals; the
// benchmark stream records the timer's start event and then launches the last arrival, so the
// measured interval starts at the simultaneous release and ends when every domain's stream is done.
// Co-residency of the spinning gates is guaranteed by the domains' disjoint SM sets.
//
// Usage (see bench/radix_sort/keys_shaped.cu):
//   shaped_lanes lanes{str_to_launch_shape(state.get_string("Shape")), state};  // skips if not possible
//   lane_buffers<T> in = make_lane_buffers(lanes, whole_input);                  // per-domain copies for 2L
//   exec_shaped(state, lanes, [&](int lane, cudaStream_t s, caching_allocator_t& a) { ... env(s, a) ... });

#pragma once

#include <cuda/__driver/driver_api.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvbench_helper.cuh>

enum class launch_shape
{
  whole,
  domains_local,
  domains_interleaved
};

inline launch_shape str_to_launch_shape(const std::string& s)
{
  if (s == "W")
  {
    return launch_shape::whole;
  }
  if (s == "2L" || s == "L")
  {
    return launch_shape::domains_local;
  }
  if (s == "2NL" || s == "NL")
  {
    return launch_shape::domains_interleaved;
  }
  throw std::runtime_error("unknown launch shape: " + s + " (W, 2L, 2NL)");
}

namespace launch_shape_detail
{
__device__ __forceinline__ unsigned long long gtimer()
{
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// One block per domain stream: arrive, then spin until `expected` arrivals (the domains plus the
// benchmark stream's release). A 5 s timeout breaks the spin instead of hanging the sweep.
__global__ void gate_kernel(unsigned* arrivals, unsigned expected)
{
  if (threadIdx.x == 0)
  {
    __threadfence();
    atomicAdd(arrivals, 1u);
    const unsigned long long t0 = gtimer();
    while (*reinterpret_cast<volatile unsigned*>(arrivals) < expected)
    {
      __nanosleep(100);
      if (gtimer() - t0 > 5000000000ull)
      {
        break;
      }
    }
  }
}

__global__ void arrive_kernel(unsigned* arrivals)
{
  atomicAdd(arrivals, 1u);
}
} // namespace launch_shape_detail

// The green contexts, streams, memory pools and gate state of one shape. Constructed per benchmark
// invocation and torn down with it: nothing outlives the measurement, so nvbench's cudaDeviceReset at
// exit finds no live green context (libcu++'s cached locality domains would, hence the driver API).
struct shaped_lanes
{
  launch_shape shape;
  int lanes = 1;
  std::vector<::CUgreenCtx> contexts; // one per locality domain (empty for `whole`)
  std::vector<::CUstream> streams; // one stream per context
  std::vector<::CUmemoryPool> pools; // each domain's default (localized) pool
  std::vector<std::unique_ptr<caching_allocator_t>> allocs; // one temp-storage allocator per lane
  unsigned* arrivals = nullptr;
  cudaEvent_t reset_done{};
  std::vector<cudaEvent_t> done;
  bool skipped = false;

  shaped_lanes(launch_shape sh, nvbench::state& state)
      : shape(sh)
  {
    if (shape != launch_shape::whole)
    {
#if _CCCL_CTK_AT_LEAST(13, 4)
      const int ordinal       = state.get_device()->get_id();
      const ::CUdevice device = ::cuda::__driver::__deviceGet(ordinal);
      const int domain_count =
        ::cuda::__driver::__deviceGetAttribute(::CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, device);
      if (domain_count < 2)
      {
        state.skip("launch shape needs >= 2 locality domains");
        skipped = true;
        return;
      }
      lanes = domain_count;
      // The same split libcu++ uses for device_ref::__locality_domains(): one SM group per domain.
      const auto full = ::cuda::__driver::__deviceGetDevResource(device, ::CU_DEV_RESOURCE_TYPE_SM);
      std::vector<::CU_DEV_SM_RESOURCE_GROUP_PARAMS> params(lanes);
      for (int k = 0; k < lanes; ++k)
      {
        params[k]                  = {};
        params[k].flags            = ::CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID;
        params[k].localityDomainId = static_cast<unsigned int>(k);
      }
      std::vector<::CUdevResource> groups(lanes);
      static_cast<void>(
        ::cuda::__driver::__devSmResourceSplit(groups.data(), static_cast<unsigned int>(lanes), full, params.data()));
      for (int k = 0; k < lanes; ++k)
      {
        const auto desc = ::cuda::__driver::__devResourceGenerateDesc(groups.data() + k, 1);
        contexts.push_back(::cuda::__driver::__greenCtxCreate(device, desc));
        streams.push_back(::cuda::__driver::__greenCtxStreamCreate(contexts.back(), ::CU_STREAM_NON_BLOCKING, 0));
        ::CUmemLocation location{};
        location.type                       = ::CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
        location.localized.deviceId         = static_cast<unsigned char>(ordinal);
        location.localized.localityDomainId = static_cast<unsigned char>(k);
        pools.push_back(::cuda::__driver::__getDefaultMemPool(location, ::CU_MEM_ALLOCATION_TYPE_PINNED));
      }
      NVBENCH_CUDA_CALL(cudaMalloc(&arrivals, sizeof(unsigned)));
      NVBENCH_CUDA_CALL(cudaEventCreateWithFlags(&reset_done, cudaEventDisableTiming));
      done.resize(lanes);
      for (auto& e : done)
      {
        NVBENCH_CUDA_CALL(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      }
#else // ^^^ CTK 13.4+ ^^^ / vvv older toolkits have no locality domains vvv
      state.skip("launch shapes need CUDA 13.4+ (locality domains)");
      skipped = true;
      return;
#endif
    }
    for (int k = 0; k < lanes; ++k)
    {
      allocs.emplace_back(std::make_unique<caching_allocator_t>());
    }
  }
  shaped_lanes(const shaped_lanes&)            = delete;
  shaped_lanes& operator=(const shaped_lanes&) = delete;
  ~shaped_lanes()
  {
    if (arrivals)
    {
      cudaDeviceSynchronize();
      cudaFree(arrivals);
      cudaEventDestroy(reset_done);
      for (auto e : done)
      {
        cudaEventDestroy(e);
      }
    }
    for (auto s : streams)
    {
      static_cast<void>(::cuda::__driver::__streamDestroyNoThrow(s));
    }
    for (auto c : contexts)
    {
      static_cast<void>(::cuda::__driver::__greenCtxDestroyNoThrow(c));
    }
  }

  [[nodiscard]] cudaStream_t stream(int lane) const
  {
    return reinterpret_cast<cudaStream_t>(streams[lane]);
  }
  // Elements per lane and the lane's element offset into a whole-device buffer.
  [[nodiscard]] std::size_t items(std::size_t total) const
  {
    return total / static_cast<std::size_t>(lanes);
  }
  [[nodiscard]] std::size_t offset(int lane, std::size_t total) const
  {
    return shape == launch_shape::domains_interleaved ? static_cast<std::size_t>(lane) * items(total) : 0;
  }
};

// Per-lane views of one logical array: for `domains_local` a copy of each lane's slice in that
// domain's memory (freed with the object), otherwise a pointer into the whole-device buffer (at the
// lane's offset for 2NL). Declare after the `shaped_lanes` it uses, so it is destroyed first.
template <typename T>
struct lane_buffers
{
  shaped_lanes* lanes = nullptr;
  std::vector<T*> ptrs;
  std::vector<T*> owned;
  lane_buffers()                               = default;
  lane_buffers(const lane_buffers&)            = delete;
  lane_buffers& operator=(const lane_buffers&) = delete;
  lane_buffers(lane_buffers&& o) noexcept
      : lanes(o.lanes)
      , ptrs(std::move(o.ptrs))
      , owned(std::move(o.owned))
  {
    o.owned.clear();
  }
  ~lane_buffers()
  {
    for (std::size_t k = 0; k < owned.size(); ++k)
    {
      static_cast<void>(
        ::cuda::__driver::__freeAsyncNoThrow(reinterpret_cast<::CUdeviceptr>(owned[k]), lanes->streams[k]));
    }
    if (!owned.empty())
    {
      cudaDeviceSynchronize();
    }
  }
  [[nodiscard]] T* ptr(int lane) const
  {
    return ptrs[lane];
  }
};

template <typename T>
lane_buffers<T> make_lane_buffers(shaped_lanes& lanes, thrust::device_vector<T>& whole)
{
  lane_buffers<T> out;
  out.lanes           = &lanes;
  const std::size_t n = whole.size();
  T* base             = thrust::raw_pointer_cast(whole.data());
  for (int k = 0; k < lanes.lanes; ++k)
  {
    if (lanes.shape == launch_shape::domains_local)
    {
      const std::size_t bytes = lanes.items(n) * sizeof(T);
      T* p = reinterpret_cast<T*>(::cuda::__driver::__mallocFromPoolAsync(bytes, lanes.pools[k], lanes.streams[k]));
      out.owned.push_back(p);
      NVBENCH_CUDA_CALL(cudaMemcpyAsync(
        p, base + static_cast<std::size_t>(k) * lanes.items(n), bytes, cudaMemcpyDeviceToDevice, lanes.stream(k)));
      out.ptrs.push_back(p);
    }
    else
    {
      out.ptrs.push_back(base + lanes.offset(k, n));
    }
  }
  if (lanes.shape == launch_shape::domains_local)
  {
    NVBENCH_CUDA_CALL(cudaDeviceSynchronize());
  }
  return out;
}

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
// cub_bench_env for an explicit stream (a lane's), same memory-resource convention.
template <typename... MoreEnvs>
auto cub_bench_env_on(caching_allocator_t& alloc, cudaStream_t stream, MoreEnvs... envs)
{
  return cuda::std::execution::env{
    cuda::stream_ref{stream},
    cuda::std::execution::prop{cuda::mr::get_memory_resource, cuda::mr::resource_ref<>{alloc}},
    envs...};
}
#endif

// Runs `body(lane, stream, alloc)` once per lane per measurement. `whole`: the plain gpu
// measurement on the benchmark stream. Domains: the gated protocol described at the top.
template <typename Body>
void exec_shaped(nvbench::state& state, shaped_lanes& lanes, Body&& body)
{
  if (lanes.skipped)
  {
    return;
  }
  if (lanes.shape == launch_shape::whole)
  {
    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
      body(0, launch.get_stream().get_stream(), *lanes.allocs[0]);
    });
    return;
  }
  const unsigned expected = static_cast<unsigned>(lanes.lanes) + 1u;
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch, auto& timer) {
    cudaStream_t s0 = launch.get_stream().get_stream();
    NVBENCH_CUDA_CALL(cudaMemsetAsync(lanes.arrivals, 0, sizeof(unsigned), s0));
    NVBENCH_CUDA_CALL(cudaEventRecord(lanes.reset_done, s0));
    // Launch-order rule: every gate before anything is enqueued behind any of them.
    for (int k = 0; k < lanes.lanes; ++k)
    {
      NVBENCH_CUDA_CALL(cudaStreamWaitEvent(lanes.stream(k), lanes.reset_done, 0));
      launch_shape_detail::gate_kernel<<<1, 32, 0, lanes.stream(k)>>>(lanes.arrivals, expected);
    }
    for (int k = 0; k < lanes.lanes; ++k)
    {
      body(k, lanes.stream(k), *lanes.allocs[k]);
      NVBENCH_CUDA_CALL(cudaEventRecord(lanes.done[k], lanes.stream(k)));
    }
    // Everything is enqueued: the timer's start event, then the arrival that releases the gates.
    timer.start();
    launch_shape_detail::arrive_kernel<<<1, 1, 0, s0>>>(lanes.arrivals);
    for (int k = 0; k < lanes.lanes; ++k)
    {
      NVBENCH_CUDA_CALL(cudaStreamWaitEvent(s0, lanes.done[k], 0));
    }
    timer.stop();
  });
}
