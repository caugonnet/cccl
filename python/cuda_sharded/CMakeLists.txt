# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.30)

# Linux-only, like cuda-stf: reject other targets before project() enables the
# CUDA language.
if (DEFINED CMAKE_SYSTEM_NAME)
  set(_cuda_sharded_target_os "${CMAKE_SYSTEM_NAME}")
else()
  set(_cuda_sharded_target_os "${CMAKE_HOST_SYSTEM_NAME}")
endif()
if (NOT _cuda_sharded_target_os STREQUAL "Linux")
  message(
    FATAL_ERROR
    "cuda-sharded is only supported on Linux (target system: ${_cuda_sharded_target_os})."
  )
endif()

# Must be set before project() initializes the CUDA language. Override with
# -DCMAKE_CUDA_ARCHITECTURES=<arch> (e.g. via the CMAKE_ARGS environment
# variable) to build for a specific GPU.
include(../../cmake/CCCLCheckCudaArchitectures.cmake)
set(
  CMAKE_CUDA_ARCHITECTURES
  "${minimum_cccl_arch}"
  CACHE STRING
  "CUDA architectures for CCCL"
)

project(
  cuda_sharded
  DESCRIPTION "Python package cuda_sharded"
  LANGUAGES CUDA CXX
)

find_package(CUDAToolkit REQUIRED)

set(CUDA_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR})
set(CUDA_VERSION_DIR "cu${CUDA_VERSION_MAJOR}")
message(
  STATUS
  "Building for CUDA ${CUDA_VERSION_MAJOR}, output directory: ${CUDA_VERSION_DIR}"
)

# Header-only consumption of the in-repo CCCL trees: no add_subdirectory of
# the parent project is needed (the extension compiles the cudax sharded
# headers directly into its shim translation units).
set(_cccl_root ${cuda_sharded_SOURCE_DIR}/../..)

# ---------------------------------------------------------------------------
# Cython toolchain (same probe as python/cuda_stf)
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

set(CYTHON_version_command "${Python3_EXECUTABLE}" -m cython --version)
execute_process(
  COMMAND ${CYTHON_version_command}
  OUTPUT_VARIABLE CYTHON_version_output
  ERROR_VARIABLE CYTHON_version_output
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

if ("${CYTHON_version_output}" MATCHES "^[Cc]ython version ([^,]+)")
  set(CYTHON_VERSION "${CMAKE_MATCH_1}")
else()
  message(
    FATAL_ERROR
    "Failed to parse Cython version from:\n${CYTHON_version_output}"
  )
endif()

# -3 generates source for Python 3
# -M generates depfile
# -t cythonizes if PYX is newer than preexisting output
# -w sets working directory
# --cplus: the bindings call C++ shim functions (exception translation)
set(
  CYTHON_FLAGS
  -3
  --cplus
  -M
  -t
  -w
  "${cuda_sharded_SOURCE_DIR}"
)

message(STATUS "Using Cython ${CYTHON_VERSION}")

# ---------------------------------------------------------------------------
# cuda.sharded._experimental extension
# ---------------------------------------------------------------------------
set(_sharded_install_dir "cuda/sharded/_experimental/${CUDA_VERSION_DIR}")

set(
  sharded_pyx_source_file
  "${cuda_sharded_SOURCE_DIR}/cuda/sharded/_experimental/_sharded_bindings_impl.pyx"
)
set(
  _sharded_generated_extension_src
  "${cuda_sharded_BINARY_DIR}/_sharded_bindings_impl.cxx"
)
set(
  _sharded_depfile
  "${cuda_sharded_BINARY_DIR}/_sharded_bindings_impl.cxx.dep"
)

add_custom_command(
  OUTPUT "${_sharded_generated_extension_src}"
  COMMAND
    "${Python3_EXECUTABLE}" -m cython
    # gersemi: off
    ${CYTHON_FLAGS}
    "${sharded_pyx_source_file}"
    --output-file "${_sharded_generated_extension_src}"
  # gersemi: on
  DEPENDS "${sharded_pyx_source_file}"
  DEPFILE "${_sharded_depfile}"
  COMMENT
    "Cythonizing ${sharded_pyx_source_file} for CUDA ${CUDA_VERSION_MAJOR}"
)

add_custom_target(
  cythonize_sharded_bindings_impl
  ALL
  DEPENDS "${_sharded_generated_extension_src}"
)

python3_add_library(
  _sharded_bindings_impl
  MODULE
  WITH_SOABI
  "${_sharded_generated_extension_src}"
  "${cuda_sharded_SOURCE_DIR}/csrc/sharded_shim.cu"
  "${cuda_sharded_SOURCE_DIR}/csrc/sharded_shim_sort.cu"
)
add_dependencies(_sharded_bindings_impl cythonize_sharded_bindings_impl)

target_include_directories(
  _sharded_bindings_impl
  PRIVATE
    "${cuda_sharded_SOURCE_DIR}/csrc"
    "${_cccl_root}/libcudacxx/include"
    "${_cccl_root}/cub"
    "${_cccl_root}/thrust"
    "${_cccl_root}/cudax/include"
)

set_target_properties(
  _sharded_bindings_impl
  PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

target_compile_options(
  _sharded_bindings_impl
  PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda>"
)

# cudart_static keeps the wheel free of a libcudart.so runtime dependency;
# the driver library provides the VMM entry points used by the contiguous
# (localized) backing.
target_link_libraries(
  _sharded_bindings_impl
  PRIVATE CUDA::cudart_static CUDA::cuda_driver
)

install(TARGETS _sharded_bindings_impl DESTINATION "${_sharded_install_dir}")
