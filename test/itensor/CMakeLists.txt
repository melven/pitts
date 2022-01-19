# Look for itensor
find_package(ITensor)

if(NOT ITensor_FOUND)
  message(STATUS "ITensor was not found, skipping ITensor related tests.")

  set(ITENSOR_TESTS PARENT_SCOPE)
else()
  message(STATUS "ITensor found:")
  message(STATUS "  library location: ${ITENSOR_LIBRARY}")
  message(STATUS "  include directory: ${ITENSOR_INCLUDE_DIR}")

  file(GLOB SRCUNITTESTS *.cpp *.hpp)
  add_library(obj_itensor_tests OBJECT EXCLUDE_FROM_ALL ${SRCUNITTESTS})
  target_include_directories(obj_itensor_tests PRIVATE ${ITENSOR_INCLUDE_DIRS} ${GTEST_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(obj_itensor_tests PRIVATE "-DGTEST_HAS_MPI=1")
  target_link_libraries(obj_itensor_tests PUBLIC ${ITENSOR_LIBRARIES})
  add_dependencies(obj_itensor_tests GTest)

  set(ITENSOR_TESTS obj_itensor_tests PARENT_SCOPE)
endif()
