# Look for itensor
find_library(ITENSOR_LIBRARY itensor)
if(NOT ITENSOR_LIBRARY)
  message(STATUS "ITensor was not found, skipping ITensor related tests.")

  set(ITENSOR_TESTS PARENT_SCOPE)
else()
  get_filename_component(ITENSOR_LIBRARY_DIR ${ITENSOR_LIBRARY} DIRECTORY CACHE)
  find_path(ITENSOR_INCLUDE_DIR itensor/all.h HINTS ${ITENSOR_LIBRARY_DIR}/.. REQUIRED)
  message(STATUS "ITensor found:")
  message(STATUS "  library location: ${ITENSOR_LIBRARY}")
  message(STATUS "  include directory: ${ITENSOR_INCLUDE_DIR}")

  # we also need hdf5 and threading support
  find_package(HDF5 COMPONENTS C HL REQUIRED)
  find_package(OpenMP REQUIRED)

  file(GLOB SRCUNITTESTS *.cpp *.hpp)
  add_library(obj_itensor_tests OBJECT EXCLUDE_FROM_ALL ${SRCUNITTESTS})
  target_include_directories(obj_itensor_tests PRIVATE ${ITENSOR_INCLUDE_DIR} ${GTEST_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(obj_itensor_tests PRIVATE "-DGTEST_HAS_MPI=1")
  target_link_libraries(obj_itensor_tests PUBLIC ${ITENSOR_LIBRARY} hdf5::hdf5_hl hdf5::hdf5 OpenMP::OpenMP_CXX)
  add_dependencies(obj_itensor_tests GTest)

  set(ITENSOR_TESTS obj_itensor_tests PARENT_SCOPE)
endif()
