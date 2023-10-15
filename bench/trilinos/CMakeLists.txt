# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.18.0)

project(tsqr CXX)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math")

add_executable(tsqr tsqr.cpp)
foreach(pkg Tpetra Teuchos)
  find_package(${pkg} REQUIRED)
  target_include_directories(tsqr PUBLIC ${${pkg}_INCLUDE_DIRS})
  #target_include_directories(tsqr PUBLIC ${${pkg}_TPL_INCLUDE_DIRS})
  target_link_libraries(tsqr PUBLIC ${${pkg}_LIBRARIES})
  #target_link_libraries(tsqr PUBLIC ${${pkg}_TPL_LIBRARIES})
endforeach()

find_package(OpenMP REQUIRED)
target_link_libraries(tsqr PUBLIC OpenMP::OpenMP_CXX)

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
target_link_libraries(tsqr PUBLIC MPI::MPI_CXX)
