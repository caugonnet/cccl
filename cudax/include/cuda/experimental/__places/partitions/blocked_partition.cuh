//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Definition of the `blocked_partition` strategy
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__places/partitions/cyclic_shape.cuh>
#include <cuda/experimental/__places/partition_recipe_builders.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>

namespace cuda::experimental::places
{
template <::std::ptrdiff_t which_dim = -1>
class blocked_partition_custom
{
public:
  blocked_partition_custom() = default;

  [[nodiscard]] static partition_recipe recipe()
  {
    return make_blocked_partition_recipe(which_dim);
  }

  template <size_t dimensions>
  _CCCL_HOST_DEVICE static auto apply(const box<dimensions>& in, pos4 place_position, dim4 grid_dims)
  {
    ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, dimensions> bounds;
    const size_t target_dim = __partition_detail::__normalize_axis(which_dim, dimensions);

    //        if constexpr (dimensions > 1) {
    for (size_t d = 0; d < dimensions; d++)
    {
      // First position in this dimension (included)
      bounds[d].first = in.get_begin(d);
      // Last position in this dimension (excluded)
      bounds[d].second = in.get_end(d);
    }
    //        }

    const size_t nplaces       = ::std::max<size_t>(grid_dims.x, 1);
    ::std::ptrdiff_t dim_beg   = bounds[target_dim].first;
    ::std::ptrdiff_t dim_end   = bounds[target_dim].second;
    const size_t cnt           = static_cast<size_t>(dim_end - dim_beg);
    const ::std::ptrdiff_t part_size =
      static_cast<::std::ptrdiff_t>(__partition_detail::__blocked_part_size(cnt, nplaces));

    // If first = second, this means it's an empty shape. This may happen
    // when there are more entries in grid_dims than in the shape for
    // example
    bounds[target_dim].first  = ::std::min(dim_beg + part_size * place_position.x, dim_end);
    bounds[target_dim].second = ::std::min(dim_beg + part_size * (place_position.x + 1), dim_end);

    return box(bounds);
  }

  template <typename mdspan_shape_t>
  _CCCL_HOST_DEVICE static auto apply(const mdspan_shape_t& in, pos4 place_position, dim4 grid_dims)
  {
    constexpr size_t dimensions = mdspan_shape_t::rank();

    ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, dimensions> bounds;
    for (size_t d = 0; d < dimensions; d++)
    {
      bounds[d].first  = 0;
      bounds[d].second = in.extent(d);
    }

    // Last position in this dimension (excluded)
    const size_t target_dim = __partition_detail::__normalize_axis(which_dim, dimensions);

    ::std::ptrdiff_t dim_end = in.extent(target_dim);

    // The last dimension is split across the different places
    const size_t nplaces      = ::std::max<size_t>(grid_dims.x, 1);
    const size_t part_size    = __partition_detail::__blocked_part_size(in.extent(target_dim), nplaces);
    bounds[target_dim].first  = ::std::min((::std::ptrdiff_t) part_size * place_position.x, dim_end);
    bounds[target_dim].second = ::std::min((::std::ptrdiff_t) part_size * (place_position.x + 1), dim_end);

    return box<dimensions>(bounds);
  }

  _CCCL_HOST_DEVICE static void get_executor(pos4* result, pos4 data_coords, dim4 data_dims, dim4 grid_dims)
  {
    const size_t rank       = data_dims.get_rank() + 1;
    const size_t target_dim = __partition_detail::__normalize_axis(which_dim, rank);
    const size_t extent     = data_dims.get(target_dim);

    const size_t nplaces   = ::std::max<size_t>(grid_dims.x, 1);
    const size_t part_size = __partition_detail::__blocked_part_size(extent, nplaces);

    // Get the coordinate in the selected dimension
    const size_t c = static_cast<size_t>(data_coords.get(target_dim));

    *result = pos4(::std::min(c / part_size, nplaces - 1));
  }
};

//! Partitions a multidimensional box or shape into contiguous blocks along a selected dimension.
//!
//! This partitioning strategy divides the data space into contiguous blocks, distributing them
//! across execution places. By default, partitioning occurs along the last dimension, but a
//! specific dimension can be selected using the template parameter. This approach ensures
//! good spatial locality and is particularly effective for regular data access patterns.
using blocked_partition = blocked_partition_custom<>;

#ifdef UNITTESTED_FILE
UNITTEST("blocked partition with very large data arrays")
{
  // blocked_partition splits along a single dimension (the highest-rank one
  // by default) and maps to grid_dims.x places.

  // 400 x 300 x 200 x 1000 = 24,000,000,000 elements (~24 billion)
  dim4 massive_4d_dims(400, 300, 200, 1000);
  EXPECT(massive_4d_dims.size() == 24000000000ULL);
  EXPECT(massive_4d_dims.size() > (1ULL << 34));

  // Partition the t-dimension (rank 3, extent 1000) into 4 blocks
  dim4 grid_dims(4);

  pos4 first_coord(0, 0, 0, 0);
  pos4 middle_coord(200, 150, 100, 500);
  pos4 last_coord(399, 299, 199, 999);

  pos4 first_pos, middle_pos, last_pos;
  blocked_partition::get_executor(&first_pos, first_coord, massive_4d_dims, grid_dims);
  blocked_partition::get_executor(&middle_pos, middle_coord, massive_4d_dims, grid_dims);
  blocked_partition::get_executor(&last_pos, last_coord, massive_4d_dims, grid_dims);

  // part_size = ceil(1000/4) = 250
  // t=0   -> block 0, t=500 -> block 2, t=999 -> block 3
  EXPECT(first_pos.x == 0);
  EXPECT(middle_pos.x == 2);
  EXPECT(last_pos.x == 3);
};

#endif // UNITTESTED_FILE
} // namespace cuda::experimental::places
