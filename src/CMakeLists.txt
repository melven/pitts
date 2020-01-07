add_library(pitts INTERFACE)
target_sources(pitts INTERFACE
  pitts_chunk.hpp
  pitts_tensor2.hpp
  pitts_tensor2_random.hpp
  pitts_tensor2_eigen_adaptor.hpp
  pitts_tensor2_qb_decomposition.hpp
  pitts_tensor3.hpp
  pitts_tensor3_random.hpp
  pitts_tensortrain.hpp
  pitts_tensortrain_norm.hpp
  pitts_tensortrain_dot.hpp
  pitts_tensortrain_random.hpp
  pitts_tensortrain_normalize.hpp
  pitts_tensortrain_axpby.hpp
  pitts_tensortrain_laplace_operator.hpp
  pitts_fixed_tensor3.hpp
  pitts_fixed_tensor3_random.hpp
  pitts_fixed_tensor3_apply.hpp
  pitts_fixed_tensor3_combine.hpp
  pitts_fixed_tensor3_split.hpp
  pitts_fixed_tensortrain_random.hpp
  pitts_fixed_tensortrain_dot.hpp
  pitts_fixed_tensortrain_axpby.hpp
  )

install(TARGETS pitts PUBLIC_HEADER DESTINATION include)

# test executables
add_executable(norm2_bench norm2_bench.cpp)
add_executable(dot_bench dot_bench.cpp)
add_executable(normalize_bench normalize_bench.cpp)
add_executable(axpby_bench axpby_bench.cpp)
add_executable(cg_bench cg_bench.cpp)
add_executable(qubit_simulator_bench qubit_simulator_bench.cpp)

# Python binding
if(pybind11_FOUND)
  pybind11_add_module(pitts_py MODULE pitts_qubit_simulator_pybind.cpp)
  install(TARGETS pitts_py LIBRARY DESTINATION lib)
endif()
