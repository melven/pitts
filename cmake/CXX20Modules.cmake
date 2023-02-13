option(PITTS_USE_MODULES "Use C++20 modules instead of header-only code" Off)
if(PITTS_USE_MODULES)
  add_definitions(-DPITTS_USE_MODULES)
  # we can try this for compatibility with pybind11
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  #add_definitions(-D_FILE_OFFSET_BITS=64)
endif()

if(PITTS_USE_MODULES)
  set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "2182bf5c-ef0d-489a-91da-49dbc3090d2a")
  set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)

  # ======================================== GCC SETTINGS ========================================
  if( CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmodules-ts")
    string(CONCAT CMAKE_EXPERIMENTAL_CXX_SCANDEP_SOURCE
      "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -E -x c++ <SOURCE>"
      " -MT <DYNDEP_FILE> -MD -MF <DEP_FILE>"
      " -fdep-file=<DYNDEP_FILE> -fdep-output=<OBJECT> -fdep-format=p1689r5"
      " -o <PREPROCESSED_SOURCE>")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FORMAT "gcc")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FLAG "-fmodule-mapper=<MODULE_MAP_FILE> -fdep-format=p1689r5 -x c++")
  endif()
  # ====================================== GCC SETTINGS END ======================================

  # ======================================= CLANG SETTINGS =======================================
  if( CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    get_filename_component(CLANG_PREFIX ${CMAKE_CXX_COMPILER} DIRECTORY)
    find_program(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS clang-scan-deps HINTS ${CLANG_PREFIX})
    string(CONCAT CMAKE_EXPERIMENTAL_CXX_SCANDEP_SOURCE
      "${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}"
      " -format=p1689 -- "
      " <CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -x c++ <SOURCE>"
      " -MT <DYNDEP_FILE> -MD -o <OBJECT>"
      " > <DYNDEP_FILE>")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FORMAT "clang")
    set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FLAG "@<MODULE_MAP_FILE>")
  endif()
  # ===================================== CLANG SETTINGS END =====================================
endif()
