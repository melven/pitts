# CMake helper script to extract optparse

include(ExternalProject)
ExternalProject_Add(
    optparse
    URL ${CMAKE_SOURCE_DIR}/third-party/optparse.tar.gz
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/optparse
    EXCLUDE_FROM_ALL 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
# variables required by the user
set(OPTPARSE_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/optparse)
