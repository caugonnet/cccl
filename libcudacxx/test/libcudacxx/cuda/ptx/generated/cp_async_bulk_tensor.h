// This file was automatically generated. Do not edit.

// We use a special strategy to force the generation of the PTX. This is mainly
// a fight against dead-code-elimination in the NVVM layer.
//
// The reason we need this strategy is because certain older versions of ptxas
// segfault when a non-sensical sequence of PTX is generated. So instead, we try
// to force the instantiation and compilation to PTX of all the overloads of the
// PTX wrapping functions.
//
// We do this by writing a function pointer of each overload to the kernel
// parameter `fn_ptr`.
//
// Because `fn_ptr` is possibly visible outside this translation unit, the
// compiler must compile all the functions which are stored.

__global__ void test_cp_async_bulk_tensor(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_global_t, void*, const void*, const int32_t (&)[1], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_shared_t, cuda::ptx::space_global_t, void*, const void*, const int32_t (&)[1], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [dstMem],
        // [tensorMap, tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const int32_t (&)[1],
                               uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));
          // cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [dstMem],
          // [tensorMap, tensorCoords], [smem_bar];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t,
                                   cuda::ptx::space_global_t,
                                   cuda::ptx::cta_group_2_t,
                                   void*,
                                   const void*,
                                   const int32_t (&)[1],
                                   uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_101a,
    (
        // cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [dstMem],
        // [tensorMap, tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const int32_t (&)[1],
                               uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));
          // cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [dstMem],
          // [tensorMap, tensorCoords], [smem_bar];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t,
                                   cuda::ptx::space_global_t,
                                   cuda::ptx::cta_group_2_t,
                                   void*,
                                   const void*,
                                   const int32_t (&)[1],
                                   uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, const void*, const int32_t (&)[1], const void*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_global_t, void*, const void*, const int32_t (&)[2], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_shared_t, cuda::ptx::space_global_t, void*, const void*, const int32_t (&)[2], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [dstMem],
        // [tensorMap, tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const int32_t (&)[2],
                               uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));
          // cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [dstMem],
          // [tensorMap, tensorCoords], [smem_bar];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t,
                                   cuda::ptx::space_global_t,
                                   cuda::ptx::cta_group_2_t,
                                   void*,
                                   const void*,
                                   const int32_t (&)[2],
                                   uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_101a,
    (
        // cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [dstMem],
        // [tensorMap, tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const int32_t (&)[2],
                               uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));
          // cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [dstMem],
          // [tensorMap, tensorCoords], [smem_bar];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t,
                                   cuda::ptx::space_global_t,
                                   cuda::ptx::cta_group_2_t,
                                   void*,
                                   const void*,
                                   const int32_t (&)[2],
                                   uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, const void*, const int32_t (&)[2], const void*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_global_t, void*, const void*, const int32_t (&)[3], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_shared_t, cuda::ptx::space_global_t, void*, const void*, const int32_t (&)[3], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [dstMem],
        // [tensorMap, tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const int32_t (&)[3],
                               uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));
          // cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [dstMem],
          // [tensorMap, tensorCoords], [smem_bar];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t,
                                   cuda::ptx::space_global_t,
                                   cuda::ptx::cta_group_2_t,
                                   void*,
                                   const void*,
                                   const int32_t (&)[3],
                                   uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_101a,
    (
        // cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [dstMem],
        // [tensorMap, tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const int32_t (&)[3],
                               uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));
          // cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [dstMem],
          // [tensorMap, tensorCoords], [smem_bar];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t,
                                   cuda::ptx::space_global_t,
                                   cuda::ptx::cta_group_2_t,
                                   void*,
                                   const void*,
                                   const int32_t (&)[3],
                                   uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, const void*, const int32_t (&)[3], const void*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_global_t, void*, const void*, const int32_t (&)[4], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_shared_t, cuda::ptx::space_global_t, void*, const void*, const int32_t (&)[4], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [dstMem],
        // [tensorMap, tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const int32_t (&)[4],
                               uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));
          // cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [dstMem],
          // [tensorMap, tensorCoords], [smem_bar];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t,
                                   cuda::ptx::space_global_t,
                                   cuda::ptx::cta_group_2_t,
                                   void*,
                                   const void*,
                                   const int32_t (&)[4],
                                   uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_101a,
    (
        // cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [dstMem],
        // [tensorMap, tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const int32_t (&)[4],
                               uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));
          // cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [dstMem],
          // [tensorMap, tensorCoords], [smem_bar];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t,
                                   cuda::ptx::space_global_t,
                                   cuda::ptx::cta_group_2_t,
                                   void*,
                                   const void*,
                                   const int32_t (&)[4],
                                   uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, const void*, const int32_t (&)[4], const void*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_cluster_t, cuda::ptx::space_global_t, void*, const void*, const int32_t (&)[5], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap,
        // tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_shared_t, cuda::ptx::space_global_t, void*, const void*, const int32_t (&)[5], uint64_t*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [dstMem],
        // [tensorMap, tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const int32_t (&)[5],
                               uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));
          // cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [dstMem],
          // [tensorMap, tensorCoords], [smem_bar];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t,
                                   cuda::ptx::space_global_t,
                                   cuda::ptx::cta_group_2_t,
                                   void*,
                                   const void*,
                                   const int32_t (&)[5],
                                   uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_101a,
    (
        // cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [dstMem],
        // [tensorMap, tensorCoords], [smem_bar];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t,
                               cuda::ptx::space_global_t,
                               cuda::ptx::cta_group_1_t,
                               void*,
                               const void*,
                               const int32_t (&)[5],
                               uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));
          // cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [dstMem],
          // [tensorMap, tensorCoords], [smem_bar];
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_shared_t,
                                   cuda::ptx::space_global_t,
                                   cuda::ptx::cta_group_2_t,
                                   void*,
                                   const void*,
                                   const int32_t (&)[5],
                                   uint64_t*)>(cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, cuda::ptx::space_shared_t, const void*, const int32_t (&)[5], const void*)>(
            cuda::ptx::cp_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800
}
