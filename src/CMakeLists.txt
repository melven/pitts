add_library(pitts INTERFACE)
target_sources(pitts INTERFACE
  pitts_chunk.hpp
  pitts_chunk_ops.hpp
  pitts_chunk_ops_plain.hpp
  pitts_chunk_ops_avx2.hpp
  pitts_chunk_ops_avx512.hpp
  pitts_kernel_info.hpp
  pitts_hash_function.hpp
  pitts_type_name.hpp
  pitts_scope_info.hpp
  pitts_timer.hpp
  pitts_performance.hpp
  pitts_common.hpp
  pitts_parallel.hpp
  pitts_tensor2.hpp
  pitts_tensor2_random.hpp
  pitts_tensor2_eigen_adaptor.hpp
  pitts_tensor2_qb_decomposition.hpp
  pitts_tensor3.hpp
  pitts_tensor3_random.hpp
  pitts_tensor3_combine.hpp
  pitts_tensor3_split.hpp
  pitts_tensor3_unfold.hpp
  pitts_tensor3_fold.hpp
  pitts_tensortrain.hpp
  pitts_tensortrain_norm.hpp
  pitts_tensortrain_dot.hpp
  pitts_tensortrain_random.hpp
  pitts_tensortrain_normalize.hpp
  pitts_tensortrain_axpby.hpp
  pitts_tensortrain_laplace_operator.hpp
  pitts_tensortrain_from_dense.hpp
  pitts_tensortrain_from_dense_classical.hpp
  pitts_tensortrain_from_dense_twosided.hpp
  pitts_tensortrain_to_dense.hpp
  pitts_tensortrain_from_itensor.hpp
  pitts_tensortrain_to_itensor.hpp
  pitts_tensortrain_operator.hpp
  pitts_tensortrain_operator_apply.hpp
  pitts_tensortrain_operator_apply_transposed.hpp
  pitts_tensortrain_operator_apply_op.hpp
  pitts_tensortrain_operator_apply_transposed_op.hpp
  pitts_tensortrain_operator_apply_dense.hpp
  pitts_tensortrain_operator_apply_transposed_dense.hpp
  pitts_tensortrain_operator_from_itensor.hpp
  pitts_tensortrain_operator_to_itensor.hpp
  pitts_tensortrain_solve_mals.hpp
  pitts_fixed_tensor3.hpp
  pitts_fixed_tensor3_random.hpp
  pitts_fixed_tensor3_apply.hpp
  pitts_fixed_tensor3_combine.hpp
  pitts_fixed_tensor3_split.hpp
  pitts_fixed_tensortrain_random.hpp
  pitts_fixed_tensortrain_dot.hpp
  pitts_fixed_tensortrain_axpby.hpp
  pitts_multivector.hpp
  pitts_multivector_random.hpp
  pitts_multivector_cdist.hpp
  pitts_multivector_centroids.hpp
  pitts_multivector_tsqr.hpp
  pitts_multivector_eigen_adaptor.hpp
  pitts_multivector_transform.hpp
  pitts_multivector_transpose.hpp
  pitts_multivector_reshape.hpp
  pitts_multivector_dot.hpp
  pitts_multivector_norm.hpp
  pitts_multivector_scale.hpp
  pitts_multivector_axpby.hpp
  pitts_gmres.hpp
  )

install(TARGETS pitts PUBLIC_HEADER DESTINATION include)

# test executables
add_executable(norm2_bench norm2_bench.cpp)
add_executable(dot_bench dot_bench.cpp)
add_executable(normalize_bench normalize_bench.cpp)
add_executable(axpby_bench axpby_bench.cpp)
add_executable(cg_bench cg_bench.cpp)
add_executable(qubit_simulator_bench qubit_simulator_bench.cpp)
add_executable(cdist2_bench cdist2_bench.cpp)
add_executable(centroids_bench centroids_bench.cpp)
add_executable(kmeans_bench kmeans_bench.cpp)
add_executable(tsqr_bench tsqr_bench.cpp)
add_executable(tsqr_single_bench tsqr_single_bench.cpp)
add_executable(tsmm_bench tsmm_bench.cpp)
add_executable(tsmm_single_bench tsmm_single_bench.cpp)
add_executable(transpose_bench transpose_bench.cpp)
add_executable(multivector_copy_bench multivector_copy_bench.cpp)
add_executable(tt_from_dense_classical_bench tt_from_dense_classical_bench.cpp)
add_executable(tt_from_dense_bench tt_from_dense_bench.cpp)
add_executable(tt_from_dense_twosided_bench tt_from_dense_twosided_bench.cpp)
add_executable(tt_from_dense_thickbounds_bench tt_from_dense_thickbounds_bench.cpp)
add_executable(tt_from_dense_thickbounds_single_bench tt_from_dense_thickbounds_single_bench.cpp)
add_executable(tt_from_dense_twosided_thickbounds_bench tt_from_dense_twosided_thickbounds_bench.cpp)

# Python binding
if(pybind11_FOUND)

  # additional parts that require ITensor
  if(ITensor_FOUND)
    set(PITTS_PYBIND_ITENSOR_SOURCE
      pitts_tensortrain_operator_itensor_autompo_pybind.hpp
      pitts_tensortrain_operator_itensor_autompo_pybind.cpp
      )
  else()
    set(PITTS_PYBIND_ITENSOR_SOURCE)
  endif()

  pybind11_add_module(pitts_py MODULE NO_EXTRAS
    pitts_pybind.cpp
    pitts_qubit_simulator_pybind.hpp
    pitts_qubit_simulator_pybind.cpp
    pitts_tensortrain_pybind.hpp
    pitts_tensortrain_pybind.cpp
    pitts_tensortrain_operator_pybind.hpp
    pitts_tensortrain_operator_pybind.cpp
    pitts_common_pybind.hpp
    pitts_common_pybind.cpp
    pitts_multivector_pybind.hpp
    pitts_multivector_pybind.cpp
    ${PITTS_PYBIND_ITENSOR_SOURCE}
    )

  if(ITensor_FOUND)
    target_compile_definitions(pitts_py PRIVATE -DPITTS_HAVE_ITENSOR)
    target_include_directories(pitts_py PRIVATE ${ITENSOR_INCLUDE_DIRS})
    target_link_libraries(pitts_py PRIVATE ${ITENSOR_LIBRARIES})
  endif()

  install(TARGETS pitts_py LIBRARY DESTINATION lib)
endif()
