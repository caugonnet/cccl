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

The algorithm family:

- elementwise: ``fill``, ``sequence``, ``iota``, ``tabulate``, ``generate``,
  ``for_each``, ``transform`` (in-place, unary, binary) — no cross-place
  stage;
- ``reduce`` / ``sum`` / ``min`` / ``max``: per-place CUB ``DeviceReduce``
  plus a combine of the per-place partials;
- ``inclusive_scan`` / ``exclusive_scan``: per-place CUB ``DeviceScan``, then
  per-place prefixes folded back in place;
- ``adjacent_difference``: local differences plus one boundary element per
  shard;
- ``count`` / ``count_if``: per-place CUB transform-reduce plus a sum of the
  per-place counts;
- ``histogram_even``: per-place CUB ``DeviceHistogram`` plus a per-bin sum of
  the per-place histograms;
- ``copy_if`` / ``filter`` / ``remove_if``: per-place CUB ``DeviceSelect``
  compaction in place, then shard sizes and offsets are updated;
- ``unique``: per-place CUB ``DeviceSelect::Unique`` in place, then
  duplicates straddling shard boundaries are trimmed with an O(1) size
  decrement per boundary;
- ``sort``: global in-place sort with the MGMN distributed sort as the
  engine, driven through the places communicator (see below).

Algorithm temporaries are drawn from each shard's place through the group's
per-place memory resources.

Size-mutating algorithms and the contiguous backing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``copy_if`` / ``filter`` / ``remove_if`` and ``unique`` shrink shard sizes in
place (capacities are unchanged; ``reset_sizes_to_capacity()`` reuses the
buffers). On a contiguous array this is unrepresentable: shrinking a shard
would leave a gap between its valid elements and the next shard's, falsifying
the read-as-one-array contract of ``contiguous_data()``, while compacting
across the gap would migrate elements onto other places than the caller asked
for. These algorithms therefore throw ``std::invalid_argument`` on contiguous
(``allocate_contiguous``) arrays, leaving them untouched. Read-only
algorithms (``count`` / ``count_if``, ``histogram_even``, ``reduce`` et al.)
remain available on every sharded array, contiguous ones included.

The places communicator and the MGMN bridge
-------------------------------------------

``places_communicator`` is a places-backed model of the ``__multi_gpu``
communicator concept: each place of a ``place_group`` becomes one rank
(rank = place index), and the MGMN range algorithms
(``cuda::experimental::reduce``, ``inclusive_scan``, ``sort``, ...) run over
it unmodified. Because the places of one process share a virtual address
space, the communicator verbs lower to device-to-device copies, and
``all_reduce`` to a single fold kernel that combines every rank's partial in
fixed rank order — bit-identical results run to run for a fixed place list.
A second variant, ``basic_places_communicator``, omits ``all_reduce`` so the
algorithms' gather-plus-local-combine path stays exercised too.

``bind_engine(group, data)`` is the two-tier seam: the container tier
manufactures what an MGMN engine consumes — one communicator, environment
(stream + per-place memory resource), iterator and size per shard:

.. code-block:: cpp

   auto b = bind_engine(group, data);
   cuda::experimental::reduce(
     cuda::experimental::broadcasted, b.comms, b.envs, b.shard_data, b.shard_sizes, outs);

Engine temporaries are drawn from each rank's environment, so scratch is
placed where the rank's work runs. Code written against the communicator
surface is portable across rungs: the same call runs over in-process places
here and over multi-process ranks with an NCCL-backed communicator.

sort: an MGMN construct as the engine
-------------------------------------

``sort(group, data, comp)`` sorts the logical array globally, in place. It is
the two-tier structure end to end: the container tier manufactures the
communicator/environment/iterator ranges with ``bind_engine``; the engine
tier is the ``__multi_gpu`` distributed sort running over them, unmodified.

The engine delivers to each rank its slice of the globally sorted sequence,
redistributed back to the rank's original element count — so shard sizes,
offsets and capacities are unchanged by construction, and contiguous
(``allocate_contiguous``) arrays are fully supported: after the sort,
``contiguous_data()`` reads as one globally sorted array. Sorting is not
stable. For keys-only sorting the result is unique as a multiset, so repeated
runs on the same input are byte-identical.

The engine slot is swappable behind the same name and contract: an
in-process, placement-aware specialization can replace it later as a
performance change, not an API change.
