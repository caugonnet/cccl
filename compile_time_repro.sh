#!/usr/bin/env bash
# Reproduce the compile time of cudax/examples/execution/lane_sharded_pipeline.cu.
#
#   ./compile_time_repro.sh                                   # slow variant (this branch)
#   ./compile_time_repro.sh senders/lane-scheduler            # fast variant (the PR branch), same example
#
# Clones caugonnet/cccl at the given branch into ./cccl-compile-repro-<branch>,
# configures the cudax preset for the local GPU, builds only the example target
# and reports wall time plus nvcc's per-phase timings (-time).
#
# The two variants differ in one thing: the slow one reads the memory resource
# from the sender environment (`read_env(get_memory_resource) | let_value(...)`
# at the root of the pipeline), the fast one takes it as a parameter. Everything
# below the root is the same code.
set -euo pipefail

BRANCH="${1:-senders/lane-scheduler-slow-compile-repro}"
REPO="${REPO:-https://github.com/caugonnet/cccl.git}"
WORK="${WORK:-$PWD/cccl-compile-repro-${BRANCH//\//-}}"
TARGET=cudax.example.execution.lane_sharded_pipeline

if [ ! -d "$WORK" ]; then
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$WORK"
fi
cd "$WORK"
export CCCL_BUILD_INFIX=repro
cmake --preset cudax \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -Dcudax_ENABLE_NCCL=OFF \
  -DCMAKE_CUDA_FLAGS="-time ${WORK}/nvcc_time.csv" \
  > "${WORK}/configure.log" 2>&1

echo "building ${TARGET} on branch ${BRANCH} ..."
start=$(date +%s)
ninja -C build/repro/cudax "${TARGET}" > "${WORK}/build.log" 2>&1 || { tail -30 "${WORK}/build.log"; exit 1; }
echo "wall time: $(( $(date +%s) - start )) s"
echo "nvcc phases (ms) for the example's translation unit:"
grep -i "lane_sharded_pipeline" "${WORK}/nvcc_time.csv" | awk -F, '{printf "  %-30s %10.1f s\n", $4, $5/1000}' || cat "${WORK}/nvcc_time.csv"
