# Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

# - Try to find LIKWID
# Once done this will define
#  LIKWID_FOUND - system has liblikwid
#  LIKWID_INCLUDE_DIRS - the likwid include directories
#  LIKWID_LIBRARIES - libraries needed to link with likwid
#  # LIKWID_DEFINITIONS - Compiler switches required for using likwid


# search for likwid.h
find_path(LIKWID_INCLUDE_DIR likwid.h)

# search for likwid library
find_library(LIKWID_LIBRARY likwid HINTS ${LIKWID_INCLUDE_DIR}/../lib)

# setup variables
set(LIKWID_INCLUDE_DIRS ${LIKWID_INCLUDE_DIR})
set(LIKWID_LIBRARIES ${LIKWID_LIBRARY})
#if(${CMAKE_Fortran_COMPILER_ID} STREQUAL "GNU")
#  set(LIKWID_DEFINITIONS "-pthread")
#endif()


include(FindPackageHandleStandardArgs)
#sets LIKWIND_FOUND and handle REQUIRED etc
find_package_handle_standard_args(LIKWID DEFAULT_MSG LIKWID_LIBRARY LIKWID_INCLUDE_DIR)

mark_as_advanced(LIKWID_INCLUDE_DIR LIKWID_LIBRARY)
