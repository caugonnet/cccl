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

Factory naming follows a two-word rule: ``adopt`` = zero-copy view over
caller-owned memory (the container becomes a view and the caller owes the
memory's lifetime); ``from_*`` = builds owned storage by copying or
transforming its input. ``sharded_array<T>::adopt(shards)`` is the named
form of the adopting constructor.

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
- ``sort``: global in-place sort with a swappable tier-2 engine: a
  shared-address-space engine where every shard lives on one device
  (locality domains), and the MGMN distributed sort driven through the
  places communicator otherwise (see below).

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

sort: one name, two engines
---------------------------

``sort(group, data, comp, engine)`` sorts the logical array globally, in
place. It is the two-tier structure end to end: the container tier owns
placement and bookkeeping; the tier-2 engine owns the cross-place
choreography, and there is one engine per rung of the cooperation-scope
ladder, selected by ``sort_engine``:

- ``sort_engine::shared_va`` — the places-rung engine, chosen automatically
  when every shard lives in one device's address space (locality domains, or
  the device itself). It combines through what that rung shares — direct
  loads across shard boundaries: per-place local sorts
  (``cub::DeviceRadixSort`` for arithmetic keys under the default
  ascending/descending orders, ``cub::DeviceMergeSort`` for arbitrary
  comparators), exact splitters computed by multi-sequence selection at the
  container's fixed shard boundaries, and a fused gather-merge in which each
  destination place merges the selected sub-ranges of the sorted runs
  straight into its own shard storage. Because the splitters are exact and
  the selection's ties are broken totally, the output lands on the original
  boundaries by construction and the whole pipeline is deterministic.

- ``sort_engine::distributed`` — the ranks-rung engine: the container tier
  manufactures the communicator/environment/iterator ranges with
  ``bind_engine`` and the ``__multi_gpu`` distributed sort runs over them,
  unmodified. This is the portability path (the same construct sorts across
  multi-process ranks), and the fallback wherever the shards do not share an
  address space.

``sort_engine::automatic`` (the default) picks by detection; the explicit
values pin an engine for A/B comparison or portability testing. Requesting
``shared_va`` where shards do not share one device's address space throws.

Both engines honor the same contract. Each shard ends holding the slice of
the globally sorted sequence at its original boundaries — shard sizes,
offsets and capacities are unchanged by construction, and contiguous
(``allocate_contiguous``) arrays are fully supported: after the sort,
``contiguous_data()`` reads as one globally sorted array. Sorting is not
stable. For keys-only sorting the result is unique as a multiset, so repeated
runs on the same input are byte-identical whichever engine runs. The engine
is a performance choice, not an API change.

sharded_csr and the sparse products: a closed library as the engine
-------------------------------------------------------------------

``sharded_csr<T>`` is a row-partitioned CSR sparse matrix: one shard per
place of a ``place_group``, each shard a self-contained CSR operator for a
contiguous row range (nnz slice plus offsets rebased to zero), stored in its
place's memory. Because every shard is a complete CSR matrix, a CLOSED
library that only understands pointers and a stream can consume it with one
ordinary call per shard — the container carries the placement, the library
never changes. The container is vendor-free and ships in the umbrella header.
``sharded_csr::from_device`` ingests a CSR whose arrays already live on the
device; per the ``from_*`` naming rule it builds owned storage — offsets are
rebased into container-owned shards and colinds/values are copied
device-to-device into the shards' places, so nothing aliases the caller's
arrays and they may be freed once it returns.

The cuSPARSE-backed products live in the separate opt-in header
``<cuda/experimental/sharded_sparse.cuh>``, which requires the cuSPARSE
development headers (it ``#error``\ s otherwise, like
``<cuda/experimental/cufile.cuh>``) and linking against cuSPARSE:

.. code-block:: cpp

   #include <cuda/experimental/sharded_sparse.cuh>

   auto group = place_group::by_locality_domains();
   sharded_csr<double> A(group, rows, cols, h_offsets, h_colinds, h_values);
   auto y = A.make_row_partitioned();          // disjoint row blocks, no combine
   spmv(group, A, d_x, y, alpha, beta);        // one confined call per shard
   auto C = A.make_row_partitioned(n_cols, /* contiguous */ true);
   spmm(group, A, d_B, C, n_cols);             // C readable as ONE array

Each call runs one cuSPARSE call per shard on the shard's place stream
(``cusparseSetStream``). Per-(shard, operation) library state — handle,
descriptors, workspace, preprocessed plan — is created lazily on the first
call into the container's type-erased ``lib_state()`` slots, built once
against the shard's fixed addresses, and reused for the matrix's lifetime;
subsequent calls only rebind the dense pointers when they change. The row
partition makes the output row blocks disjoint, so there is never a combine
step, and outputs compose with ``allocate_contiguous`` backings unchanged.

Dense operands (``x``, ``B``) are plain device pointers readable from every
place. Which per-place COPIES of a re-read operand should exist — and when a
write makes them stale — is a coherence question that belongs to the binding
tier: an STF ``logical_data`` can materialize and cache a per-place instance
and hand its pointer to these calls; the container deliberately does not
absorb that role.

Measured rebalance
~~~~~~~~~~~~~~~~~~

An nnz-balanced row split (the default) is not a TIME-balanced split: with
each shard confined to its place's SMs, a call finishes at max(shard time),
and skewed row-length distributions make the default split pay the full
skew. ``spmv_shard_times`` / ``spmm_shard_times`` measure each shard solo
through the exact call path of the products, and
``sharded_csr::time_balanced_boundaries`` converts one measurement into a
time-equalizing split via a piecewise-rate model:

.. code-block:: cpp

   auto times = spmm_shard_times(group, A, d_B, C, n_cols);
   auto bounds = sharded_csr<double>::time_balanced_boundaries(
     rows, h_offsets, A.interior_boundaries(), times);
   sharded_csr<double> A2(group, rows, cols, h_offsets, h_colinds, h_values, bounds);

One calibration round is amortized over every subsequent call on the rebuilt
matrix — the natural fit for iterative consumers that reuse one operator
across many products. Rates shift as rows change shards, so repeat the round
(keeping the best measured split) when the skew is extreme.
