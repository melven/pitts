# Look for itensor
find_package(ITensor)

if(NOT ITensor_FOUND)
  message(STATUS "ITensor was not found, skipping ITensor related tests.")

  set(ITENSOR_TESTS PARENT_SCOPE)
  set(ITENSOR_PY_UNITTESTS PARENT_SCOPE)
else()
  file(GLOB SRCUNITTESTS *.cpp *.hpp)
  add_library(obj_itensor_tests OBJECT EXCLUDE_FROM_ALL ${SRCUNITTESTS})
  target_include_directories(obj_itensor_tests PRIVATE ${ITENSOR_INCLUDE_DIRS} ${GTEST_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(obj_itensor_tests PRIVATE "-DGTEST_HAS_MPI=1")
  target_link_libraries(obj_itensor_tests PUBLIC ${ITENSOR_LIBRARIES})
  add_dependencies(obj_itensor_tests GTest)

  # ITensor + python unittests
  file(GLOB PY_UNITTESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
  list(TRANSFORM PY_UNITTESTS PREPEND "itensor/")

  set(ITENSOR_TESTS obj_itensor_tests PARENT_SCOPE)
  set(ITENSOR_PY_UNITTESTS ${PY_UNITTESTS} PARENT_SCOPE)
endif()
