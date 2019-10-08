# CMake helper script to extract googletest

set(GTEST_VERSION "1.8.0-mpi")

include(ExternalProject)
ExternalProject_Add(
    GTest
    URL ${CMAKE_SOURCE_DIR}/third-party/gtest-${GTEST_VERSION}.tar.gz
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/gtest
    EXCLUDE_FROM_ALL 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
# variables required by the user
set(GTEST_INCLUDE_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(GTEST_SOURCES
  ${CMAKE_CURRENT_BINARY_DIR}/gtest/gtest.h
  ${CMAKE_CURRENT_BINARY_DIR}/gtest/gtest-spi.h
  ${CMAKE_CURRENT_BINARY_DIR}/gtest/gtest-all.cc
  ${CMAKE_CURRENT_BINARY_DIR}/gtest/gtest_main.cc
  )
# tell cmake that the sources are only available after extracting
foreach(gsrc_ ${GTEST_SOURCES})
  add_custom_command(OUTPUT ${gsrc_} COMMAND ; DEPENDS GTest)
endforeach()
