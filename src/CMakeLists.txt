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
  )

# test executables
add_executable(norm2_bench norm2_bench.cpp)
add_executable(dot_bench dot_bench.cpp)
