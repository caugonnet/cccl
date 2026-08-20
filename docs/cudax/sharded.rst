.. _cudax-sharded:

Sharded containers and algorithms
=================================

.. contents::
   :depth: 2

Sharded containers partition one logical array across the places of a single
process — devices, or sub-device locality domains — while keeping a common
address space. They extend the cooperation-scope structure CUDA algorithms
already follow: a primitive at one scope runs the previous scope's primitive
locally and combines results using what the new scope shares (registers and
shuffles within a warp, shared memory within a block, global memory within a
device). At the places scope, what is shared is one virtual address space with
placed pages; at the multi-process/multi-node scope, where nothing is shared,
communicator-based algorithms take over
(see the MGMN algorithms built on ``__multi_gpu``).

The sharded API lives in the ``cuda::experimental::sharded`` namespace and is
available through the ``cuda/experimental/sharded.cuh`` header. It builds on
the standalone :ref:`places <cudax-places>` layer; execution resources come
from a :ref:`place_group <places-place-group>`.

sharded_array
-------------

``sharded_array<T>`` is a 1D array whose shards each carry their own
placement: a ``data_place``, an ``exec_place`` and a reference stream.

.. code-block:: cpp

   using namespace cuda::experimental::sharded;

   auto group = place_group::by_locality_domains();
   auto data  = sharded_array<double>::allocate(group, n);

   iota(group, data, 0.0);
   double total = sum(group, data);   // per-place CUB + combine

``allocate_contiguous`` places the shards inside ONE contiguous virtual
address range (VMM-backed via ``localized_array``): logical shard boundaries
are exact, physical ownership snaps to the allocation granularity, and
``contiguous_data()`` hands the whole array to unmodified single-pointer
consumers. Because the range is mapped once, shard sizes are fixed;
size-mutating operations must refuse such arrays.

Algorithms
----------

The initial algorithm family:

- elementwise: ``fill``, ``sequence``, ``iota``, ``tabulate``, ``generate``,
  ``for_each``, ``transform`` (in-place, unary, binary) — no cross-place
  stage;
- ``reduce`` / ``sum`` / ``min`` / ``max``: per-place CUB ``DeviceReduce``
  plus a combine of the per-place partials;
- ``inclusive_scan`` / ``exclusive_scan``: per-place CUB ``DeviceScan``, then
  per-place prefixes folded back in place;
- ``adjacent_difference``: local differences plus one boundary element per
  shard.

Algorithm temporaries are drawn from each shard's place through the group's
per-place memory resources.
