# Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

###
### Minimal cmake version
###
cmake_minimum_required(VERSION 3.18.0) # first to provide LAPACK::LAPACK (before it is not available in modern form)
cmake_policy(VERSION 3.18.0...3.26)


###
### General project settings
###
project(pitts LANGUAGES CXX)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS Off)
#set(CMAKE_LINK_WHAT_YOU_USE On)


###
### Use our own cmake modules
###
list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")


###
### Find and print version through git
###
find_package(Git)
mark_as_advanced(CLEAR GIT_EXECUTABLE)
if(NOT GIT_FOUND)
  message(WARNING "Could not find git; please specify the full path to the GIT_EXECUTABLE to obtain compile time version string")
endif()
include(cmake/GetGitRevisionDescription.cmake)
git_describe_working_tree(PITTS_GIT_VERSION "--tags")# "--match 'v[0-9].*'")
message(STATUS "Git version: ${PITTS_GIT_VERSION}")


###
### Compiler recognition and flags
###
option(PITTS_OPTIMIZE_FOR_HOST "Enable compiler optimization for current CPU architecture. Resulting executables might not run on other computers!" On)
if( CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(GCC_INLINE_FLAGS "--param inline-unit-growth=200 --param large-function-growth=1000 --param early-inlining-insns=2000 --param max-inline-insns-single=200 --param max-early-inliner-iterations=50 --param large-function-growth=1000 --param large-function-insns=2700")
  set(GCC_VARTRACK_FLAGS "--param max-gcse-memory=13107200  --param large-stack-frame=1024 --param max-hoist-depth=300 --param max-vartrack-size=0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts -ffast-math -Wall -Wextra -Wno-unused -Wno-sign-compare ${GCC_INLINE_FLAGS} ${GCC_VARTRACK_FLAGS}")# -fverbose-asm")
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
  if(PITTS_OPTIMIZE_FOR_HOST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mprefer-vector-width=512")
  endif()
elseif( CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Wall -Wextra -Wno-unused -Wno-sign-compare")
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
  if(PITTS_OPTIMIZE_FOR_HOST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mprefer-vector-width=512")
  endif()
elseif( CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  if(PITTS_OPTIMIZE_FOR_HOST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHOST")
  endif()
else()
  message(WARNING "Unknown CXX compiler ID '${CMAKE_CXX_COMPILER_ID}', cannot determine correct flags!")
endif()
# Fix Intellisense in VsCode if CMAKE_CXX_COMPILER_ARG1 is set (e.g. -stdlib=libc++)
if( CMAKE_CXX_COMPILER_ARG1 )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_COMPILER_ARG1} ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_COMPILER_ARG1 "")
endif()
# Speed up build time for developing
option(PITTS_DEVELOP_BUILD "Enable development build (speeds up rebuilding with small changes through external template definitions).")
if(PITTS_DEVELOP_BUILD)
  add_definitions(-DPITTS_DEVELOP_BUILD)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
endif()
# use non-optimized axpby algorithm for performance comparison
option(PITTS_TENSORTRAIN_PLAIN_AXPBY "Use slower standard algorithm for evaluating tensor-train axpby operations (disables axpby_normalized).")
mark_as_advanced(PITTS_TENSORTRAIN_PLAIN_AXPBY)
if(PITTS_TENSORTRAIN_PLAIN_AXPBY)
  add_definitions(-DPITTS_TENSORTRAIN_PLAIN_AXPBY)
endif()
# use standard QB/SVD instead of Q-less TSQR
option(PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB "Use slower standard algorithm for normalization (QR/SVD) not exploiting our Q-less TSQR implementation.")
mark_as_advanced(PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB)
if(PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB)
  add_definitions(-DPITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB)
endif()
# use standard / slower contraction scheme instead of optimized padded for TTOp * dense
option(PITTS_TENSORTRAIN_MALS_SLOWCONTRACT "Use slower standard contraction scheme for applying the local TT operator in the inner (dense) GMRES.")
mark_as_advanced(PITTS_TENSORTRAIN_MALS_SLOWCONTRACT)
if(PITTS_TENSORTRAIN_MALS_SLOWCONTRACT)
  add_definitions(-DPITTS_TENSORTRAIN_MALS_SLOWCONTRACT)
endif()
# use faster direct MKL calls in some places (Eigen->MKL somehow is slower)
option(PITTS_DIRECT_MKL_GEMM "Directly call MKL gemm in some places where this is faster than calling MKL through Eigen.")
if(PITTS_DIRECT_MKL_GEMM)
  add_definitions(-DPITTS_DIRECT_MKL_GEMM)
endif()


###
### Required libraries
###
# OpenMP
find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)

find_package(MPI REQUIRED)
link_libraries(MPI::MPI_CXX)

# LAPACK
find_package(LAPACK REQUIRED)
link_libraries(LAPACK::LAPACK)

# Eigen 3 library (for linear algebra calculations)
find_package (Eigen3 3.3.7 REQUIRED NO_MODULE)
link_libraries(Eigen3::Eigen)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_NO_AUTOMATIC_RESIZING -DEIGEN_USE_BLAS")
# Option for possibly faster/parallel SVD algorithm with MKL, not enabled per default as it seems to cause problems sometimes...
option(PITTS_EIGEN_USE_LAPACKE "Define EIGEN_USE_LAPACKE to call LAPACK routines for e.g. singular value computations" False)
if(PITTS_EIGEN_USE_LAPACKE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_LAPACKE -DEIGEN_USE_MKL")
endif()

# pybind11 for the (optional) python binding...
find_package(pybind11)
if(pybind11_FOUND)
  message(STATUS "Found pybind11, building the python interface...")
else()
  message(STATUS "Did not find pybind11, not building the python interface...")
endif()

# libcxxwrap-julia for the (optional) julia binding
find_package(JlCxx)
if(JlCxx_FOUND)
  message(STATUS "Found JlCxx (libcxxwrap-julia), building the Julia interface...")
  # this is overwritten by JlCxx, so set it again!
  set(CMAKE_CXX_STANDARD_REQUIRED On)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_EXTENSIONS Off)
else()
  message(STATUS "Did not find JlCxx (libcxxwrap-julia), not building the Julia interface...")
endif()

# cereal for for data (de-)serialization
find_package(cereal REQUIRED NO_MODULE)
# old cereal versions provide cereal, newer cereal::cereal
if(TARGET cereal)
  link_libraries(cereal)
else()
  link_libraries(cereal::cereal)
endif()

# likwid for performance measurements and system information
find_package(LIKWID)
if(LIKWID_FOUND)
  message(STATUS "Using likwid to detect CPU cache sizes.")
  include_directories(${LIKWID_INCLUDE_DIRS})
  link_libraries(${LIKWID_LIBRARIES})
  add_definitions(-DPITTS_HAVE_LIKWID)
else()
  message(WARNING "likwid was not found - using hardcoded values for CPU cache sizes.")
endif()
option(PITTS_USE_LIKWID_MARKER_API "Uses the likwid-perfctr marker api for performance measurements." False)
if(PITTS_USE_LIKWID_MARKER_API)
  if(NOT LIKWID_FOUND)
    message(FATAL_ERROR "Option PITTS_USE_LIKWID_MARKER_API set but likwid was not found!")
  endif()
  add_definitions(-DPITTS_USE_LIKWID_MARKER_API -DLIKWID_PERFMON)
endif()


# ITensor for easy setup of tensor train operators from quantum physics
find_package(ITensor)
if(ITensor_FOUND)
  message(STATUS "Found ITensor, building ITensor support.")
  message(STATUS "  library location: ${ITENSOR_LIBRARY}")
  message(STATUS "  include directory: ${ITENSOR_INCLUDE_DIR}")
else()
  message(STATUS "Did not find ITensor, not building ITensor support.")
endif()


###
### Define source files and executables
###
add_subdirectory(src)


###
### Define test files
###
add_subdirectory(test)
