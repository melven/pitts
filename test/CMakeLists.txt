enable_testing()

# get googletest for C++ unit tests
include(GTest)

# allow code coverage
include(CodeCoverage)

# files for C++ unit tests
file(GLOB_RECURSE CPP_SRCUNITTESTS *.cpp *.hpp)

# executable for the unit tests
add_executable(pitts_tests EXCLUDE_FROM_ALL ${CPP_SRCUNITTESTS} ${GTEST_SOURCES})
target_include_directories(pitts_tests PRIVATE ${GTEST_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(pitts_tests PRIVATE "-DGTEST_HAS_MPI=1")

# add the unit tests
add_test(NAME unittests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND $<TARGET_FILE:pitts_tests> --gtest_output=xml:pitts_tests.xml)
set_tests_properties(unittests PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=12;OMPI_MCA_hwloc_base_binding_policy=none")

# tests for the python binding
file(GLOB_RECURSE PY_UNITTESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
if(pybind11_FOUND)
  foreach(tc ${PY_UNITTESTS})
    add_test(NAME ${tc} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pitts_py>:$ENV{PYTHONPATH}
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${tc} -v)
  endforeach()
endif()

# useful target to (re)build and test
add_custom_target (check
                   COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> -V --timeout 300
                   SOURCES CMakeLists.txt ${PY_UNITTESTS})
add_dependencies(check pitts_tests)
if( CODE_COVERAGE )
  setup_target_for_coverage(check)
endif()
