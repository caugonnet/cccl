# Compile-time reproducer: lane_sharded_pipeline

cudax/examples/execution/lane_sharded_pipeline.cu in two variants that differ only
in how the memory resource reaches the verbs (slow_vs_fast.diff):

- fast: passed as a parameter (`start(x, mr)`), ~1 min to compile;
- slow: read from the sender environment at the root of the pipeline,
  `read_env(get_memory_resource) | let_value(...)`, >10 min.

Measured with ./compile_time_repro.sh on an RTX 3080 Ti host (gcc 13, CUDA 13.0,
fresh clone, -arch=native, -O3, -std=c++17), nvcc -time per phase:

    phase               fast      slow
    cudafe++           4.9 s     6.5 s
    cicc             130.8 s   785.2 s   <- device-side front end (EDG)
    gcc (compiling)    6.7 s     8.6 s
    wall              145 s     803 s

The 6x is entirely in cicc, i.e. template instantiation in the device pass, not
in GCC's optimizer. The only change is that one read_env at the root makes the
whole pipeline below it environment-dependent.

How much more template work is there? Measured on the host pass:

- GCC 13 -ftime-report:  template instantiation 2.20 s -> 3.78 s (1.7x),
  overload resolution 1.83 s -> 3.42 s (1.9x), memory 307 MB -> 545 MB.
  GCC's host pass as a whole: 6.7 s -> 8.6 s.
- clang 20 -ftime-trace (host pass, fast variant only): 4,588 class and 8,595
  function template instantiations, 1,205 template families, 7.8 s front end,
  no family above 0.4 s self time. clang's device front end: 6 s (cicc: 131 s).
- clang REJECTS the slow variant: when_all.cuh:105, "member access into
  incomplete type __state_t". when_all's state computes its children's
  completion signatures in a static constexpr member inside its own class
  definition, through an environment whose `query` spells its constraint and
  return type as env_of_t<__rcvr_t> = decltype(get_env(__state_.__rcvr_)):
  with read_env children this reaches into the class being defined. EDG and GCC
  accept it; EDG pays ~6x, GCC ~1.8x, clang refuses.

With the when_all environment fixed (commit 44f23115a5 on senders/lane-scheduler:
__env_t holds the receiver and stop token, not the state), clang accepts the slow
variant and counts, host pass, fixed header:

    clang 20 -ftime-trace               fast     slow   ratio
    class instantiations                4588     5790   1.26
    function instantiations             8595    15374   1.79
    distinct template families          1205     1207   1.00
    front-end time                      8.1 s   12.7 s  1.57

  Growth is concentrated in the domain-dispatch / completion-signature machinery,
  re-run per sender per environment: get_completion_domain_t 758 -> 1731,
  __transform_sender_t 461 -> 1117, get_completion_scheduler_t 460 -> 1116,
  __lane::domain::transform_sender 389 -> 1015, __upon_t (then) 585 -> 1081,
  connect_t::__get_declfn 168 -> 412, get_completion_signatures 135 -> 329.

The fix did NOT reduce cicc's time (1128 s on the full slow variant in one run):
the ill-formedness and the cost are separate. Scaling of cicc time on the slow
variant (same code, varying the pipeline):

    3 shards, transforms = 0 / 1 / 2 / 3 :   8 s /  10 s /  77 s / 785 s
    3 transforms, shards = 1 / 2 / 3     :  19 s / 233 s / 785 s

  (runs partly concurrent, so +-30% noise; the 3/3 point was 785 s alone and
  1128 s in another run). Both dimensions compound: roughly x8-10 per added
  `then` layer and more than linear in when_all arity, while clang's
  instantiation count for the full pipeline grows only 1.8x. EDG re-derives the
  dependent domain/transform_sender/completion-signature chain at every nesting
  level without memoizing it; clang and gcc do not. Both variants need the CCCL headers of branch
senders/lane-scheduler (caugonnet/cccl); the slow one is committed on branch
senders/lane-scheduler-slow-compile-repro.

Clone-and-build reproducer (configures the cudax preset for the local GPU, builds
only the example target, prints wall time and nvcc's -time phases):

    ./compile_time_repro.sh                              # slow
    ./compile_time_repro.sh senders/lane-scheduler       # fast

A single self-contained preprocessed file is not possible with nvcc: it
auto-includes cuda_runtime.h before compiling, and the expanded include guards in
`nvcc -E` output make every system header define twice.
