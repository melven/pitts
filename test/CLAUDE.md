<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
SPDX-License-Identifier: BSD-3-Clause
-->

# test/ — unit tests

See the root `CLAUDE.md` for how to build and run them (`ctest` in `build/test`, environment variables).

## Structure

- **C++ tests** (`test_*.cpp`) use googletest 1.8.0 with an MPI patch (`third-party/gtest-1.8.0-mpi.tar.gz`, unpacked by `cmake/GTest.cmake`). All `.cpp` files are globbed into one executable `pitts_tests` (target is `EXCLUDE_FROM_ALL`, so build it explicitly). It is run with 1 and 3 MPI processes (`PITTS_TEST_NPROCS`) and `PITTS_TEST_NTHREADS` OpenMP threads; some tests assert `omp_get_max_threads() >= 4` or run only on a given process count.
- Test suite names follow `PITTS_<Class>_<topic>`, typed tests iterate over float/double/complex. There is roughly one `test_<x>.cpp` per `src/pitts_<x>.hpp`.
- **Python tests** (`test_*_pybind.py`, `unittest`) test the `pitts_py` module and are only registered when pybind11 was found. They run through `mpiexec -n 1` with `PYTHONPATH` pointing to the built module.
- **Julia tests** (`test_*_jlbind.jl`) are only registered when JlCxx is found (not available here).
- `itensor/`: tests needing ITensor (MPS/MPO conversions, AutoMPO binding); compiled into `pitts_tests` only when ITensor is found (see `itensor/CMakeLists.txt`).
- New `.cpp`/`.py`/`.jl` files in this directory are picked up automatically by the `file(GLOB ...)` in `CMakeLists.txt` (re-run cmake).

## Helper headers

| File | Content |
|---|---|
| `eigen_test_helper.hpp` | `ASSERT_NEAR`/`EXPECT_NEAR` overloads for Eigen matrices (element-wise comparison with pretty printing). |
| `test_complex_helper.hpp` | `ASSERT_NEAR` overloads for `std::complex`. |
| `test_chunk_ops.hpp` | Typed test bodies for the chunk operations, parameterized via the `TEST_NAME` macro; included by `test_chunk_ops.cpp` (dispatch header), `test_chunk_ops_plain.cpp`, `test_chunk_ops_avx2.cpp` (`#ifdef __AVX2__`) and `test_chunk_ops_avx512.cpp` (`#ifdef __AVX512F__`). |

## C++ test files (by topic)

Core data structures and infrastructure:

| File | Tests |
|---|---|
| `test_chunk.cpp` | `Chunk<T>` size/alignment. |
| `test_tensor2.cpp`, `test_tensor2_random.cpp`, `test_tensor2_concat.cpp`, `test_tensor2_eigen_adaptor.cpp`, `test_tensor2_qb_decomposition.cpp`, `test_tensor2_maxvol.cpp` | Tensor2 construction/resize, randomize, block concatenation, Eigen map, SVQB-style QB decomposition (incl. rank-deficient cases), maxvol row selection (dominance property, invalid/rank-deficient input). |
| `test_tensor3.cpp`, `test_tensor3_random.cpp`, `test_tensor3_combine.cpp`, `test_tensor3_split.cpp`, `test_tensor3_fold.cpp`, `test_tensor3_unfold.cpp` | Tensor3 basics, randomize, core contraction, core splitting with truncation, fold/unfold reshapes. |
| `test_fixed_tensor3.cpp`, `test_fixed_tensor3_apply.cpp`, `test_fixed_tensor3_combine.cpp`, `test_fixed_tensor3_split.cpp`, `test_fixed_tensor3_random.cpp` | Fixed-dimension rank-3 tensor: basics, apply, combine, split, random. |
| `test_fixed_tensortrain.cpp`, `test_fixed_tensortrain_axpby.cpp`, `test_fixed_tensortrain_dot.cpp`, `test_fixed_tensortrain_random.cpp` | Fixed-dimension tensor train: basics, axpby, dot, random. |
| `test_multivector.cpp`, `test_multivector_random.cpp`, `test_multivector_eigen_adaptor.cpp`, `test_multivector_reshape.cpp` | MultiVector basics, thread-count independent randomize, Eigen map, reshape. |
| `test_multivector_axpby.cpp`, `test_multivector_dot.cpp`, `test_multivector_norm.cpp`, `test_multivector_scale.cpp`, `test_multivector_gramian.cpp`, `test_multivector_cdist.cpp`, `test_multivector_centroids.cpp` | Column-wise vector operations against Eigen reference results. |
| `test_multivector_transform.cpp`, `test_multivector_transpose.cpp`, `test_multivector_triangular_mult.cpp`, `test_multivector_triangular_solve.cpp` | TSMM (incl. reshape variants), transpose, triangular multiply/solve (many size combinations, padding edge cases). |
| `test_multivector_tsqr.cpp` | Householder building blocks (`applyReflection`, `transformBlock`, `combineTwoBlocks`) and `block_TSQR` serial / 4 threads / MPI, varying sizes, rank-deficient input. |
| `test_parallel.cpp` | `ompThreadInfo`, `mpiProcInfo`, `distribute`, MPI type mapping, `mpiGather`, `mpiCombineMaps`. |
| `test_random.cpp` | Seed generation and distributions for all scalar types. |
| `test_timer.cpp`, `test_performance.cpp`, `test_scope_info.cpp`, `test_kernel_info.cpp`, `test_hash_function.cpp`, `test_type_name.cpp` | Timing/performance statistics, scope info, flop/byte accounting types, compile-time hash, type names. |
| `test_qubit_simulator.cpp` | Qubit allocation, gates, measurement, Bell state, time evolution, expectation values. |
| `test_gmres.cpp` | Generic GMRES with a dense-expanded `TensorTrainOperator` (identity, random symmetric/non-symmetric). |

TensorTrain and TensorTrainOperator:

| File | Tests |
|---|---|
| `test_tensortrain.cpp`, `test_tensortrain_random.cpp`, `test_tensortrain_debug.cpp` | TT construction, ranks, sub-tensors, copy; randomize; debug operator overloads. |
| `test_tensortrain_norm.cpp`, `test_tensortrain_dot.cpp` | Norms and inner products vs dense reference. |
| `test_tensortrain_normalize.cpp` | left/right/full normalization, orthogonality flags, truncation error bounds, boundary ranks. |
| `test_tensortrain_axpby.cpp`, `test_tensortrain_axpby_plain.cpp`, `test_tensortrain_axpby_normalized.cpp` | TT addition variants (the normalized one has many special cases: zero vectors, rank-1, boundary ranks). |
| `test_tensortrain_from_dense.cpp`, `test_tensortrain_from_dense_classical.cpp`, `test_tensortrain_from_dense_twosided.cpp`, `test_tensortrain_to_dense.cpp` | TT-SVD variants and reconstruction, incl. MPI-distributed input and rank/tolerance options. |
| `test_tensortrain_gram_schmidt.cpp` | Gram-Schmidt orthogonalization of TT vectors. |
| `test_tensortrain_laplace_operator.cpp` | Laplace stencil application. |
| `test_tensortrain_operator.cpp`, `test_tensortrain_operator_debug.cpp` | TTOp construction, `setEye`, sub-tensors, debug operators. |
| `test_tensortrain_operator_apply.cpp`, `test_tensortrain_operator_apply_transposed.cpp`, `test_tensortrain_operator_apply_op.cpp`, `test_tensortrain_operator_apply_transposed_op.cpp`, `test_tensortrain_operator_apply_dense.cpp`, `test_tensortrain_operator_apply_transposed_dense.cpp` | All apply variants vs dense reference. |
| `test_tensortrain_operator_to_dense.cpp`, `test_tensortrain_operator_to_qtt.cpp` | Dense expansion and QTT conversion. |
| `test_tensortrain_sweep_index.cpp` | Sweep iteration for ALS/MALS/TT-GMRES widths. |
| `test_tensortrain_solve_gmres.cpp` | TT-GMRES on identity and random (symmetric/non-symmetric) systems, 1 to 6 dimensions (the `random_nDim2` case is the slowest test). |
| `test_tensortrain_solve_mals.cpp` | Largest file: ALS/MALS/AMEn variants, symmetric and Petrov-Galerkin, different dimensions and options. |
| `test_tensortrain_solve_mals_helper.cpp` | Internal helper loops of MALS. |

## Python tests

| File | Tests |
|---|---|
| `test_common_pybind.py` | `initialize`/`finalize`, statistics printing. |
| `test_multivector_pybind.py` | Creation, resize, numpy buffer protocol, copy, randomize, `cdist2`. |
| `test_tensortrain_pybind.py` | Creation, ranks, from/to dense (incl. TSQR variant), set/get sub-tensors, dot/norm/normalize/axpby/gramSchmidt. |
| `test_tensortrain_operator_pybind.py` | Creation, `setEye`, apply/applyT, axpby, operator product, toDense, toQtt. |
| `test_tensortrain_solve_pybind.py` | `solveMALS` and `solveGMRES` on identity systems. |
| `test_qubit_simulator_pybind.py` | Simple circuit example. |
| `test_ttn_local_contraction_pybind.py` | Tree-tensor-network local gradient contraction vs numpy einsum. |

## itensor/

`test_tensortrain_from_itensor.cpp`, `test_tensortrain_to_itensor.cpp`, `test_tensortrain_operator_from_itensor.cpp`, `test_tensortrain_operator_to_itensor.cpp` (MPS/MPO round trips) and `test_tensortrain_operator_itensor_autompo_pybind.py` (AutoMPO wrapper). Skipped without ITensor.
