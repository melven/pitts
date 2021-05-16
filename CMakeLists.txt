###
### Minimal cmake version
###
cmake_minimum_required(VERSION 3.14.0) # not really needed


###
### General project settings
###
project(pitts LANGUAGES CXX)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_STANDARD 20)


###
### Use our own cmake modules
###
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")


###
### Compiler recognition and flags
###
if( CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_definitions(-DGNU)
  set(GCC_INLINE_FLAGS "--param inline-unit-growth=200 --param large-function-growth=1000 --param early-inlining-insns=2000 --param max-inline-insns-single=200")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts -ffast-math -Wall -Wextra -Wno-unused -Wno-sign-compare ${GCC_INLINE_FLAGS}")# -fverbose-asm")
  option(PITTS_OPTIMIZE_FOR_HOST "Enable compiler optimization for current CPU architecture. Resulting executables might not run on other computers!" On)
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
  if(PITTS_OPTIMIZE_FOR_HOST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mprefer-vector-width=512")
  endif()
elseif( CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  add_definitions(-DINTEL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHOST")
else()
  message(WARNING "Unknown CXX compiler ID '${CMAKE_CXX_COMPILER_ID}', cannot determine correct flags!")
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
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DMKL_DIRECT_CALL")

# Eigen 3 library (for linear algebra calculations)
find_package (Eigen3 3.3.7 REQUIRED NO_MODULE)
link_libraries(Eigen3::Eigen)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_NO_AUTOMATIC_RESIZING -DEIGEN_DONT_PARALLELIZE")
# Option for possibly faster/parallel SVD algorithm with MKL, not enabled per default as it seems to cause problems sometimes...
option(PITTS_EIGEN_USE_LAPACKE "Define EIGEN_USE_LAPACKE to call LAPACK routines for e.g. singular value computations" False)
if(PITTS_EIGEN_USE_LAPACKE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_LAPACKE")
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
link_libraries(cereal)

# likwid for performance measurements
option(PITTS_USE_LIKWID_MARKER_API "Uses the likwid-perfctr marker api for performance measurements." False)
if(PITTS_USE_LIKWID_MARKER_API)
  find_package(LIKWID REQUIRED)
  include_directories(${LIKWID_INCLUDE_DIRS})
  link_libraries(${LIKWID_LIBRARIES})
  add_definitions(-DPITTS_USE_LIKWID_MARKER_API -DLIKWID_PERFMON)
endif()


###
### Define source files and executables
###
add_subdirectory(src)


###
### Define test files
###
add_subdirectory(test)
