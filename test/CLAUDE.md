<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
SPDX-License-Identifier: BSD-3-Clause
-->

# test/ — unit tests

See the root `CLAUDE.md` for how to build and run them (`ctest` in `build/test`, environment variables).

## Structure

- **C++ tests** (`test_*.cpp`) use googletest 1.8.0 with an MPI patch (`third-party/gtest-1.8.0-mpi.tar.gz`, unpacked by `cmake/GTest.cmake`). All `.cpp` files are globbed into one executable `pitts_tests` (target is `EXCLUDE_FROM_ALL`, so build it explicitly). It is run with 1 and 3 MPI processes (`PITTS_TEST_NPROCS`) and `PITTS_TEST_NTHREADS` OpenMP threads; some tests assert `omp_get_max_threads() >= 4` or run only on a given process count.
- In the MPI-patched googletest **every assertion is a collective operation** (`AssertionResult::boolIdenticalOnMPIprocs` does two `MPI_Allreduce`, it also verifies that the outcome is the same on all processes). A test must therefore execute the same number of assertions on every process - make the *arguments* rank-dependent, never the control flow around `EXPECT_*`, otherwise the test deadlocks with `PITTS_TEST_NPROCS > 1`.
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
| `test_tensor2.cpp`, `test_tensor2_random.cpp`, `test_tensor2_concat.cpp`, `test_tensor2_eigen_adaptor.cpp`, `test_tensor2_qb_decomposition.cpp`, `test_tensor2_maxvol.cpp` | Tensor2 construction/resize, copy to/from strided array-like objects, randomize, block concatenation, Eigen map, SVQB-style QB decomposition (incl. rank-deficient cases), maxvol row selection (dominance property, invalid/rank-deficient input, independent of the number of threads) and with oversampling (greedy volume criterion, also with several rows per block and groups of similar rows, least-squares coefficients, conditioning of the selected rows). |
| `test_tensor3.cpp`, `test_tensor3_random.cpp`, `test_tensor3_combine.cpp`, `test_tensor3_split.cpp`, `test_tensor3_fold.cpp`, `test_tensor3_unfold.cpp` | Tensor3 basics (also copy to/from strided array-like objects), randomize, core contraction, core splitting with truncation, fold/unfold reshapes. |
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
| `test_tensortrain_cross.cpp` | `applyElementwise` (TT-cross driver): argument checks, exact recovery of the identity and of x^2, sqrt / 1/x against the dense reference (1-site and 2-site variants, QTT-like dimensions with rank growth), interpolation property on the returned index sets, enrichment options (random and derived error index sets), array functor vs. scalar functor, oversampling (least-squares interpolation; also a Gram-matrix kernel where the square variant stagnates), convergence measures (scale invariance, no convergence while the ranks grow, error estimate only with enrichment), control sample (off by default, extra evaluations, detects a deliberately bad approximation), `maxRank` (limits the ranks and the oversampled index sets, also via a small neighboring dimension, 1-site and 2-site), one and two dimensions (1-site and 2-site), error index sets from the frame and random ones for rank-1 inputs (frame smaller than errorRank), one and two inputs, with and without oversampling; functions of two inputs (argument checks, exact product and quotient against the dense reference, f(x, x) = x^2, rank growth beyond the initial crosses, array functor). |
| `test_tensortrain_cross_symmetric.cpp` | `applyElementwiseSymmetric` against dense references: exact identity and square (one to three cores of the half, 1-site and 2-site), an exponential kernel of a Gram matrix (oversampling, control sample, 2-site without error crosses), the product of two symmetric inputs, maxRank, f = 0, the reported negative part of an indefinite function, argument checks. |
| `test_tensortrain_cross_helper.cpp` | Building blocks of TT-cross: nested index sets (comparison, decoding, expansion, consistency with the Tensor3 unfoldings), partial contractions and interface matrices via `SweepData` against direct evaluation of a random TT, local block assembly (1-site and 2-site) and block helpers, mixed crosses of two index set families with changing sets (the error crosses of the driver) against a direct evaluation, `FunctionOnCrosses` with two inputs (all crosses, bond blocks and the control sample) against a direct evaluation, mirrored index sets, `FunctionOnCrosses` with two inputs given by the left halves of symmetric tensors H H^T (all crosses, the symmetric middle block, recomputed boundary after a changed middle set, control sample) against a direct evaluation, the consistency checks of mirrored inputs (mirrored middle sets, boundary matrices up to date), building blocks of the symmetric driver (the full left set, `selectLeftRows` (dominance, oversampling), `solveLeftInterfaceSymmetric` (square and oversampled), `directSum` (halves side by side and sum), `initialHalfFrame` (orthonormal, normalized sum of the inputs, spans the halves, independent of their norms), `leftIndexSetsFromFrame` (full-rank interfaces, oversampling, r_d = 1) and right sets with a boundary matrix, operations on halves (left-orthonormalization, inner products, right-orthonormal form), `positiveEigen` (ordering, negative part, truncation, round-off asymmetry), `symmetricRound` (orthonormal, decreasing positive weights, error bound, maxRank, a single core)), least-squares solves with the interface matrices (inverse of the interface contractions, square and oversampled), local core updates in the orthogonal gauge (orthonormal core, reconstruction, truncation, enrichment, oversampled selection, dominance of the selection with an interface matrix), index selection without cores (`dominantRows/Columns`: fixed size, oversampled, not influenced by round-off), the initial right frame of y (orthonormal, ranks of one or several inputs, independent of their norms and gauges, cancelling inputs, maxRank, invalid inputs) and the index sets from a frame (dominance, full-rank interfaces for every input, rank-deficient and dependent inputs, oversampling). |
| `test_tensortrain_laplace_operator.cpp` | Laplace stencil application. |
| `test_tensortrain_operator.cpp`, `test_tensortrain_operator_debug.cpp` | TTOp construction, `setEye`, sub-tensors, debug operators. |
| `test_tensortrain_operator_apply.cpp`, `test_tensortrain_operator_apply_transposed.cpp`, `test_tensortrain_operator_apply_op.cpp`, `test_tensortrain_operator_apply_transposed_op.cpp`, `test_tensortrain_operator_apply_dense.cpp`, `test_tensortrain_operator_apply_transposed_dense.cpp` | All apply variants vs dense reference. |
| `test_tensortrain_operator_to_dense.cpp`, `test_tensortrain_operator_to_qtt.cpp` | Dense expansion and QTT conversion. |
| `test_tensortrain_sweep_index.cpp` | Sweep iteration for ALS/MALS/TT-GMRES widths. |
| `test_tensortrain_sweep_data.cpp` | `SweepData` cache: incremental update from both sides, invalidation (ALS and MALS width), sweep access pattern. |
| `test_tensortrain_solve_gmres.cpp` | TT-GMRES on identity and random (symmetric/non-symmetric) systems, 1 to 6 dimensions (the `random_nDim2` case is the slowest test). |
| `test_tensortrain_solve_mals.cpp` | Largest file: ALS/MALS/AMEn variants, symmetric and Petrov-Galerkin, different dimensions and options. |
| `test_tensortrain_solve_mals_helper.cpp` | Internal helper loops of MALS. |

## Python tests

| File | Tests |
|---|---|
| `test_common_pybind.py` | `initialize`/`finalize` (once per process in `setUpClass`/`tearDownClass`), statistics printing, large statistics output into a pipe drained by a python thread (jupyter-style fd capture: the binding must release the GIL, a regression hangs until the ctest timeout). |
| `test_multivector_pybind.py` | Creation, resize, numpy buffer protocol, copy, randomize, `cdist2`. |
| `test_tensortrain_pybind.py` | Creation, ranks, from/to dense (incl. TSQR variant), set/get sub-tensors (also open boundary bonds, as in C++; C/Fortran order, strided views, other dtypes), dot/norm/normalize/axpby/gramSchmidt, GIL release during a long call (a counting thread must make progress). |
| `test_tensortrain_operator_pybind.py` | Creation, `setEye`, apply/applyT, axpby, operator product, toDense, toQtt. |
| `test_tensortrain_solve_pybind.py` | `solveMALS` and `solveGMRES` on identity systems. |
| `test_tensortrain_cross_pybind.py` | `applyElementwise`: numpy ufuncs, in-place functions, `numpy.vectorize`d scalar functions, float/double, options as keyword arguments (incl. oversampling), errors (wrong result size, exception in the function); two inputs (product with `numpy.multiply`, argument order and in-place result, keyword arguments, mismatching dimensions); `applyElementwiseSymmetric` with halves (exponential kernel, 2-site, float, product of two inputs, errors); `internal.symmetricRound` (exact without truncation, error bound, maxRank, float, mismatching weights). |
| `test_qubit_simulator_pybind.py` | Simple circuit example. |
| `test_ttn_local_contraction_pybind.py` | Tree-tensor-network local gradient contraction vs numpy einsum. |

## itensor/

`test_tensortrain_from_itensor.cpp`, `test_tensortrain_to_itensor.cpp`, `test_tensortrain_operator_from_itensor.cpp`, `test_tensortrain_operator_to_itensor.cpp` (MPS/MPO round trips) and `test_tensortrain_operator_itensor_autompo_pybind.py` (AutoMPO wrapper). Skipped without ITensor.
