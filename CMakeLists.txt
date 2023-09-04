###
### Minimal cmake version
###
cmake_minimum_required(VERSION 3.14.0) # not really needed
cmake_policy(VERSION 3.14.0...3.26)


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
link_libraries(${LAPACK_LIBRARIES})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}") # -DMKL_DIRECT_CALL")

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

# cereal for for data (de-)serialization
find_package(cereal REQUIRED NO_MODULE)
# old cereal versions provide cereal, newer cereal::cereal
if(TARGET cereal)
  link_libraries(cereal)
else()
  link_libraries(cereal::cereal)
endif()

# likwid for performance measurements
option(PITTS_USE_LIKWID_MARKER_API "Uses the likwid-perfctr marker api for performance measurements." False)
if(PITTS_USE_LIKWID_MARKER_API)
  find_package(LIKWID REQUIRED)
  include_directories(${LIKWID_INCLUDE_DIRS})
  link_libraries(${LIKWID_LIBRARIES})
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
