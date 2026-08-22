//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Tests for `place_group::lib_state`, the type-erased per-place
 *        library-state cache: per-(place, type) identity and laziness,
 *        isolation between groups and between types, teardown (every cached
 *        object destroyed exactly once, with the group), and survival across
 *        a group move. Vendor-free: the cached objects are probe structs, the
 *        way the cache itself is vendor-free (vendor layers such as the
 *        cuSPARSE products cache their handles through this same API).
 */

#include <cuda/experimental/__places/place_group.cuh>

using namespace cuda::experimental::places;

namespace
{
// A probe standing in for a per-place library handle: counts constructions
// and destructions, and remembers which place it was made for.
struct probe_state
{
  static int live;
  static int created;
  static int destroyed;

  size_t place_idx;

  explicit probe_state(size_t idx)
      : place_idx(idx)
  {
    created++;
    live++;
  }

  probe_state(const probe_state&)            = delete;
  probe_state& operator=(const probe_state&) = delete;

  ~probe_state()
  {
    destroyed++;
    live--;
  }
};

int probe_state::live      = 0;
int probe_state::created   = 0;
int probe_state::destroyed = 0;

// A second cached type: (place, type) keys must not collide across types.
struct other_state
{
  int tag = 7;
};

void reset_probe_counters()
{
  probe_state::live      = 0;
  probe_state::created   = 0;
  probe_state::destroyed = 0;
}

void test_identity_and_laziness()
{
  reset_probe_counters();

  place_group group = place_group::by_locality_domains();
  const size_t n    = group.size();

  // Nothing is created before first use.
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(!group.has_lib_state<probe_state>(i));
  }
  EXPECT(probe_state::created == 0);

  // First use creates; the same (place, type) always yields the SAME object.
  ::std::vector<probe_state*> first(n);
  for (size_t i = 0; i < n; i++)
  {
    first[i] = &group.lib_state<probe_state>(i, [i] {
      return new probe_state(i);
    });
    EXPECT(group.has_lib_state<probe_state>(i));
    EXPECT(first[i]->place_idx == i);
  }
  EXPECT(probe_state::created == static_cast<int>(n));

  for (size_t i = 0; i < n; i++)
  {
    auto* again = &group.lib_state<probe_state>(i, [i]() -> probe_state* {
      // Must not be invoked: the slot is already populated.
      EXPECT(false);
      return nullptr;
    });
    EXPECT(again == first[i]);
  }
  EXPECT(probe_state::created == static_cast<int>(n));

  // Different places hold different objects.
  for (size_t i = 1; i < n; i++)
  {
    EXPECT(first[i] != first[0]);
  }

  // A different type on the same place is a different slot.
  auto& other = group.lib_state<other_state>(0, [] {
    return new other_state();
  });
  EXPECT(other.tag == 7);
  EXPECT(static_cast<void*>(&other) != static_cast<void*>(first[0]));
  // ...and creating it did not disturb the probe slots.
  EXPECT(probe_state::created == static_cast<int>(n));
}

void test_group_isolation()
{
  reset_probe_counters();

  // Two groups over the same places are distinct resource scopes: their
  // caches do not share state.
  place_group a(places_from_devices({0}));
  place_group b(places_from_devices({0}));

  auto* in_a = &a.lib_state<probe_state>(0, [] {
    return new probe_state(0);
  });
  EXPECT(!b.has_lib_state<probe_state>(0));
  auto* in_b = &b.lib_state<probe_state>(0, [] {
    return new probe_state(0);
  });
  EXPECT(in_a != in_b);
  EXPECT(probe_state::created == 2);
}

void test_teardown()
{
  reset_probe_counters();

  {
    place_group group = place_group::by_locality_domains();
    for (size_t i = 0; i < group.size(); i++)
    {
      group.lib_state<probe_state>(i, [i] {
        return new probe_state(i);
      });
    }
    EXPECT(probe_state::live == static_cast<int>(group.size()));
    // Cached objects live for the whole group lifetime...
  }
  // ...and are destroyed exactly once, with the group.
  EXPECT(probe_state::live == 0);
  EXPECT(probe_state::destroyed == probe_state::created);
}

void test_move_semantics()
{
  reset_probe_counters();

  place_group g(places_from_devices({0}));
  auto* before = &g.lib_state<probe_state>(0, [] {
    return new probe_state(0);
  });

  place_group moved(::std::move(g));
  // The cached object survives the move, no re-creation, no double destroy.
  EXPECT(moved.has_lib_state<probe_state>(0));
  auto* after = &moved.lib_state<probe_state>(0, []() -> probe_state* {
    EXPECT(false);
    return nullptr;
  });
  EXPECT(after == before);
  EXPECT(probe_state::created == 1);
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  test_identity_and_laziness();
  test_group_isolation();
  test_teardown();
  test_move_semantics();

  // Global balance: everything created was destroyed by group teardown.
  EXPECT(probe_state::live == 0);

  return 0;
}
