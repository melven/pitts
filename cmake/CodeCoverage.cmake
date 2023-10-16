# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

# integration build with code coverage in debug mode...
if( CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_BUILD_TYPE MATCHES "Debug" )
  option(CODE_COVERAGE "Enable unit test code coverage" Off)
  if(CODE_COVERAGE)
    find_program(GCOV_EXECUTABLE gcov)
    find_program(GCOVR_SCRIPT gcovr)
    #find_package(PythonInterp)
    set(GCOVR_OUTPUT_FORMAT "xml" CACHE STRING "Output format for code coverage generation with gcovr (xml html)")
    set_property(CACHE GCOVR_OUTPUT_FORMAT PROPERTY STRINGS xml html)
    set(GCOVR_OPTIONS -j 24 -r ${CMAKE_SOURCE_DIR} --filter=${PROJECT_SOURCE_DIR}/src.* --object-directory=${CMAKE_BINARY_DIR} --delete --print-summary --gcov-executable=${GCOV_EXECUTABLE}
      CACHE STRING "Options for code coverage generation with gcovr")
    set(GCOVR_OUTPUT_OPTIONS --xml coverage.xml --html coverage.html --html-details)
    mark_as_advanced(GCOV_EXECUTABLE)
    mark_as_advanced(GCOVR_SCRIPT)
    mark_as_advanced(GCOVR_OUTPUT_FORMAT)
    mark_as_advanced(GCOVR_OPTIONS)
    if(NOT GCOV_EXECUTABLE)
      message(FATAL_ERROR "Could not find gcov executable, required for CODE_COVERAGE!")
    endif()
    #if(NOT PYTHONINTERP_FOUND)
    #  message(FATAL_ERROR "Could not find python interpreter, required for CODE_COVERAGE!")
    #endif()
    if(NOT GCOVR_SCRIPT)
      message(FATAL_ERROR "Could not find gcovr script (installable with 'python -m pip install gcovr'), required for INTERATION_BUILD")
    endif()
    # create a coverage report for an existing target
    function(setup_target_for_coverage _targetname)
      # add coverage report as a post-build-step
      add_custom_command(TARGET ${_targetname} POST_BUILD
                         COMMAND ${CMAKE_SOURCE_DIR}/cmake/prepare_gcovr.sh
                         COMMAND ${GCOVR_SCRIPT} ${GCOVR_OPTIONS} ${GCOVR_OUTPUT_OPTIONS}
                         COMMENT "Generating code coverage report..."
                         VERBATIM)
    endfunction()
    # add required compile options
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-dir=coverage.%p/")
  endif()
endif()
