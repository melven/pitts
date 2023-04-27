enable_testing()

# get googletest for C++ unit tests
include(GTest)

# allow code coverage
include(CodeCoverage)

# files for C++ unit tests
file(GLOB CPP_SRCUNITTESTS *.cpp *.hpp)

# ITensor adapters
add_subdirectory(itensor)

# speed up compilation times in dev builds
if(PITTS_DEVELOP_BUILD)
  link_libraries($<TARGET_OBJECTS:pitts_impl>)
endif()

# executable for the unit tests
add_executable(pitts_tests EXCLUDE_FROM_ALL ${CPP_SRCUNITTESTS} ${GTEST_SOURCES})
target_include_directories(pitts_tests PRIVATE ${GTEST_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(pitts_tests PRIVATE "-DGTEST_HAS_MPI=1")
target_link_libraries(pitts_tests PRIVATE ${ITENSOR_TESTS})


#enable valgrind for memory checking of tests
option(PITTS_TESTS_ENABLE_VALGRIND "Run tests through valgrind for memory checking..." Off)
if(PITTS_TESTS_ENABLE_VALGRIND)
  find_program(VALGRIND_EXECUTABLE valgrind)
  set(MEMORYCHECK_COMMAND "${VALGRIND_EXECUTABLE}" "--error-exitcode=1")# "--suppressions=${PROJECT_SOURCE_DIR}/third-party/valgrind-python.supp")
else()
  set(MEMORYCHECK_COMMAND )
endif()


# allow to define the number of threads and processes used to run the tests
set(PITTS_TEST_NTHREADS "4" CACHE STRING "Number of OpenMP threads used to run the tests")
set(PITTS_TEST_NPROCS "1;3" CACHE STRING "Number of MPI processes used to run the tests")
# check if we need to use SLURM srun instead of mpiexec
find_program(SRUN_EXECUTABLE srun DOC "Executable for launching MPI parallel programs with SLURM")
if(SRUN_EXECUTABLE)
  # with slurm
  set(PITTS_TEST_MPIRUN_EXECUTABLE "${SRUN_EXECUTABLE}")
  set(PITTS_TEST_MPIRUN_PREFLAGS "-c ${PITTS_TEST_NTHREADS}" "--cpu-bind=verbose,core" "--time=00:05:00")
  set(PITTS_TEST_MPIRUN_NUMPROG_FLAG "-n")
  set(PITTS_TEST_MPIRUN_POSTFLAGS "")
else()
  # no slurm
  # set # threads and prevent unintended pinning behavior of openmpi
  set(PITTS_TEST_MPIRUN_EXECUTABLE "${MPIEXEC_EXECUTABLE}")
  set(PITTS_TEST_MPIRUN_PREFLAGS "${MPIEXEC_PREFLAGS}")
  set(PITTS_TEST_MPIRUN_NUMPROG_FLAG "${MPIEXEC_NUMPROC_FLAG}")
  set(PITTS_TEST_MPIRUN_POSTFLAGS "${MPIEXEC_POSTFLAGS}")
endif()


# add the unit tests
foreach(NPROCS ${PITTS_TEST_NPROCS})
  set(TEST_NAME "unittests_np${NPROCS}")
  set(TEST_COMMAND ${PITTS_TEST_MPIRUN_EXECUTABLE} ${PITTS_TEST_MPIRUN_PREFLAGS} ${PITTS_TEST_MPIRUN_NUMPROG_FLAG} ${NPROCS} ${MEMORYCHECK_COMMAND} $<TARGET_FILE:pitts_tests> ${PITTS_TEST_MPIRUN_POSTFLAGS} --gtest_output=xml:pitts_tests_np${NPROCS}.xml)
  add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND ${TEST_COMMAND})
  set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=${PITTS_TEST_NTHREADS};OMPI_MCA_hwloc_base_binding_policy=none;MKL_CBWR=COMPATIBLE")
  set(TEST_NAME)
  set(TEST_COMMAND)
endforeach()

# tests for the python binding
file(GLOB PY_UNITTESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
if(pybind11_FOUND)
  foreach(tc ${PY_UNITTESTS} ${ITENSOR_PY_UNITTESTS})
    add_test(NAME ${tc} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pitts_py>:$ENV{PYTHONPATH}
      ${PITTS_TEST_MPIRUN_EXECUTABLE} ${PITTS_TEST_MPIRUN_PREFLAGS} ${PITTS_TEST_MPIRUN_NUMPROG_FLAG} 1
      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${tc} -v)
    set_tests_properties(${tc} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=${PITTS_TEST_NTHREADS};OMPI_MCA_hwloc_base_binding_policy=none")
  endforeach()
endif()

# useful target to (re)build and test
add_custom_target (check
                   COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> -V --timeout 400 --output-on-failure
                   SOURCES CMakeLists.txt ${PY_UNITTESTS})
add_dependencies(check pitts_tests)
if( CODE_COVERAGE )
  setup_target_for_coverage(check)
endif()
