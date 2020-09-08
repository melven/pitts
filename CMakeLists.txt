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
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


###
### Compiler recognition and flags
###
if( CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_definitions(-DGNU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts -march=native -ffast-math -mprefer-vector-width=512 -Wall -Wextra -Wno-unused -Wno-sign-compare")# -fverbose-asm")
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_NO_AUTOMATIC_RESIZING -DEIGEN_DONT_PARALLELIZE -DEIGEN_USE_LAPACKE")# -DEIGEN_USE_MKL_ALL")

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


###
### Define source files and executables
###
add_subdirectory(src)


###
### Define test files
###
add_subdirectory(test)
