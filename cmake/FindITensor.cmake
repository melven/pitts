# Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
# SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
#
# SPDX-License-Identifier: BSD-3-Clause

# - Try to find ITensor (C++)
# Once done this will define
#  ITENSOR_FOUND - system has libitensor
#  ITENSOR_INCLUDE_DIRS - the itensor include directories
#  ITENSOR_LIBRARIES - libraries needed to link with itensor


find_library(ITENSOR_LIBRARY itensor)

if(ITENSOR_LIBRARY)
  get_filename_component(ITENSOR_LIBRARY_DIR ${ITENSOR_LIBRARY} DIRECTORY CACHE)
  find_path(ITENSOR_INCLUDE_DIR itensor/all.h HINTS ${ITENSOR_LIBRARY_DIR}/.. REQUIRED)

  # we also need hdf5 and threading support
  find_package(HDF5 COMPONENTS C HL REQUIRED)
  find_package(OpenMP REQUIRED)

  # setup variables
  set(ITENSOR_INCLUDE_DIRS ${ITENSOR_INCLUDE_DIR})
  set(ITENSOR_LIBRARIES ${ITENSOR_LIBRARY} hdf5::hdf5_hl hdf5::hdf5 OpenMP::OpenMP_CXX)

endif()

include(FindPackageHandleStandardArgs)
#sets ITENSOR_FOUND and handle REQUIRED etc
find_package_handle_standard_args(ITensor DEFAULT_MSG ITENSOR_LIBRARY ITENSOR_INCLUDE_DIR)

mark_as_advanced(ITENSOR_INCLUDE_DIR ITENSOR_LIBRARY ITENSOR_LIBRARY_DIR)
