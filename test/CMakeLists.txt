# get googletest for C++ unittests
include(GTest)

# files for C++ unittests
file(GLOB_RECURSE CPP_SRCUNITTESTS *.cpp *.hpp)
add_executable(pitts_tests EXCLUDE_FROM_ALL ${CPP_SRCUNITTESTS} ${GTEST_SOURCES})
target_include_directories(pitts_tests PRIVATE ${GTEST_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/src)

