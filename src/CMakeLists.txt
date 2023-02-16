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
  pitts_tensortrain_axpby_normalized.hpp
  pitts_tensortrain_axpby.hpp
  pitts_tensortrain_laplace_operator.hpp
  pitts_tensortrain_from_dense.hpp
  pitts_tensortrain_from_dense_classical.hpp
  pitts_tensortrain_from_dense_twosided.hpp
  pitts_tensortrain_to_dense.hpp
  pitts_tensortrain_debug.hpp
  pitts_tensortrain_from_itensor.hpp
  pitts_tensortrain_to_itensor.hpp
  pitts_tensortrain_gram_schmidt.hpp
  pitts_tensortrain_solve_gmres.hpp
  pitts_tensortrain_operator.hpp
  pitts_tensortrain_operator_apply.hpp
  pitts_tensortrain_operator_apply_transposed.hpp
  pitts_tensortrain_operator_apply_op.hpp
  pitts_tensortrain_operator_apply_transposed_op.hpp
  pitts_tensortrain_operator_apply_dense.hpp
  pitts_tensortrain_operator_apply_transposed_dense.hpp
  pitts_tensortrain_operator_from_itensor.hpp
  pitts_tensortrain_operator_to_itensor.hpp
  pitts_tensortrain_operator_debug.hpp
  pitts_tensortrain_sweep_index.hpp
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

# support for C++20 modules
if(PITTS_USE_MODULES)
  add_library(pitts_modules STATIC)
  set_target_properties(pitts_modules PROPERTIES POSITION_INDEPENDENT_CODE On)
  target_sources(pitts_modules PUBLIC FILE_SET CXX_MODULES FILES
    pitts_hash_function.cppm
    pitts_timer.cppm
    pitts_type_name.cppm
    pitts_scope_info.cppm
    pitts_kernel_info.cppm
    pitts_common.cppm
    pitts_parallel.cppm
    pitts_performance.cppm
    pitts_chunk.cppm
    pitts_chunk_ops.cppm
    pitts_fixed_tensor3.cppm
    pitts_fixed_tensor3_apply.cppm
    pitts_fixed_tensor3_combine.cppm
    pitts_fixed_tensor3_random.cppm
    pitts_fixed_tensor3_split.cppm
    pitts_fixed_tensortrain.cppm
    pitts_fixed_tensortrain_axpby.cppm
    pitts_fixed_tensortrain_dot.cppm
    pitts_fixed_tensortrain_random.cppm
    pitts_qubit_simulator.cppm
    pitts_tensor2.cppm
    pitts_tensor2_random.cppm
    pitts_tensor2_qb_decomposition.cppm
    pitts_tensor3.cppm
    pitts_tensor3_random.cppm
    pitts_tensor3_combine.cppm
    pitts_tensor3_fold.cppm
    pitts_tensor3_unfold.cppm
    pitts_tensor3_split.cppm
    pitts_tensortrain.cppm
    pitts_tensortrain_axpby.cppm
    pitts_tensortrain_axpby_normalized.cppm
    pitts_tensortrain_axpby_plain.cppm
    pitts_tensortrain_debug.cppm
    pitts_tensortrain_dot.cppm
    pitts_tensortrain_norm.cppm
    pitts_tensortrain_from_dense.cppm
    pitts_tensortrain_from_dense_classical.cppm
    pitts_tensortrain_from_dense_twosided.cppm
    pitts_tensortrain_gram_schmidt.cppm
    pitts_tensortrain_laplace_operator.cppm
    pitts_tensortrain_normalize.cppm
    pitts_multivector.cppm
    pitts_multivector_cdist.cppm
    pitts_multivector_centroids.cppm
    pitts_multivector_random.cppm
    pitts_multivector_reshape.cppm
    pitts_multivector_transform.cppm
    pitts_multivector_transpose.cppm
    pitts_multivector_tsqr.cppm
    pitts_multivector_axpby.cppm
    pitts_multivector_norm.cppm
    pitts_multivector_scale.cppm
    pitts_multivector_dot.cppm
    pitts_gmres.cppm
    )
endif()


# test executables
add_executable(axpby_left_ortho_bench axpby_left_ortho_bench.cpp)
add_executable(axpby_right_ortho_bench axpby_right_ortho_bench.cpp)
add_executable(norm2_bench norm2_bench.cpp)
add_executable(dot_bench dot_bench.cpp)
add_executable(normalize_bench normalize_bench.cpp)
add_executable(axpby_bench axpby_bench.cpp)
add_executable(apply_op_bench apply_op_bench.cpp)
add_executable(cg_bench cg_bench.cpp)
add_executable(qubit_simulator_bench qubit_simulator_bench.cpp)
add_executable(cdist2_bench cdist2_bench.cpp)
add_executable(centroids_bench centroids_bench.cpp)
add_executable(kmeans_bench kmeans_bench.cpp)
add_executable(tsqr_bench tsqr_bench.cpp)
add_executable(tsqr_single_bench tsqr_single_bench.cpp)
add_executable(tsmm_bench tsmm_bench.cpp)
add_executable(normalize_qb_bench normalize_qb_bench.cpp)
add_executable(tsmm_single_bench tsmm_single_bench.cpp)
add_executable(transpose_bench transpose_bench.cpp)
add_executable(multivector_copy_bench multivector_copy_bench.cpp)
add_executable(tt_from_dense_classical_bench tt_from_dense_classical_bench.cpp)
add_executable(tt_from_dense_bench tt_from_dense_bench.cpp)
add_executable(tt_from_dense_twosided_bench tt_from_dense_twosided_bench.cpp)
add_executable(tt_from_dense_thickbounds_bench tt_from_dense_thickbounds_bench.cpp)
add_executable(tt_from_dense_thickbounds_single_bench tt_from_dense_thickbounds_single_bench.cpp)
add_executable(tt_from_dense_twosided_thickbounds_bench tt_from_dense_twosided_thickbounds_bench.cpp)
if(PITTS_USE_MODULES)
  target_link_libraries(axpby_left_ortho_bench PRIVATE pitts_modules)
  target_link_libraries(axpby_right_ortho_bench PRIVATE pitts_modules)
  target_link_libraries(norm2_bench PRIVATE pitts_modules)
  target_link_libraries(dot_bench PRIVATE pitts_modules)
  target_link_libraries(normalize_bench PRIVATE pitts_modules)
  target_link_libraries(axpby_bench PRIVATE pitts_modules)
  target_link_libraries(apply_op_bench PRIVATE pitts_modules)
  target_link_libraries(cg_bench PRIVATE pitts_modules)
  target_link_libraries(qubit_simulator_bench PRIVATE pitts_modules)
  target_link_libraries(cdist2_bench PRIVATE pitts_modules)
  target_link_libraries(centroids_bench PRIVATE pitts_modules)
  target_link_libraries(kmeans_bench PRIVATE pitts_modules)
  target_link_libraries(tsqr_bench PRIVATE pitts_modules)
  target_link_libraries(tsqr_single_bench PRIVATE pitts_modules)
  target_link_libraries(tsmm_bench PRIVATE pitts_modules)
  target_link_libraries(normalize_qb_bench PRIVATE pitts_modules)
  target_link_libraries(tsmm_single_bench PRIVATE pitts_modules)
  target_link_libraries(transpose_bench PRIVATE pitts_modules)
  target_link_libraries(multivector_copy_bench PRIVATE pitts_modules)
  target_link_libraries(tt_from_dense_classical_bench PRIVATE pitts_modules)
  target_link_libraries(tt_from_dense_bench PRIVATE pitts_modules)
  target_link_libraries(tt_from_dense_twosided_bench PRIVATE pitts_modules)
  target_link_libraries(tt_from_dense_thickbounds_bench PRIVATE pitts_modules)
  target_link_libraries(tt_from_dense_thickbounds_single_bench PRIVATE pitts_modules)
  target_link_libraries(tt_from_dense_twosided_thickbounds_bench PRIVATE pitts_modules)
endif()

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
    pitts_tensortrain_solve_pybind.hpp
    pitts_tensortrain_solve_pybind.cpp
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

  if(PITTS_USE_MODULES)
    target_link_libraries(pitts_py PRIVATE pitts_modules)
  endif()

  install(TARGETS pitts_py LIBRARY DESTINATION lib)
endif()
