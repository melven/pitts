<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
SPDX-License-Identifier: BSD-3-Clause
-->

# src/ — library sources, bindings and benchmark programs

All library code lives in namespace `PITTS` (internals in `PITTS::internal`). Header-only, C++20.

## File naming conventions

| Pattern | Meaning |
|---|---|
| `pitts_<topic>.hpp` | Public interface: declarations + Doxygen docs. Include this from user code. |
| `pitts_<topic>_impl.hpp` | Template implementation of the corresponding `.hpp`. Included automatically unless `PITTS_DEVELOP_BUILD` is set. |
| `pitts_<topic>.cpp` | Only compiled with `PITTS_DEVELOP_BUILD=On`: `#include`s the `_impl.hpp` and lists explicit template instantiations (float, double, complex<float>, complex<double>) into object library `pitts_impl`. Errors out if compiled without that flag. When adding a template function used from the bindings/tests, add its instantiation here. |
| `pitts_<topic>_pybind.{hpp,cpp}` | pybind11 binding (`PITTS::pybind::init_<Topic>(m)`), assembled in `pitts_pybind.cpp` into module `pitts_py`. |
| `pitts_<topic>_jlbind.{hpp,cpp}` | Julia binding via JlCxx (`PITTS::jlbind::define_<Topic>(mod)`), assembled in `pitts_jlbind.cpp` into `libpitts_jl`; only built when JlCxx is found. `PittsJl.jl` is the Julia module wrapper. |
| `*_bench.cpp`, `tsqr_parallel.cpp` | Stand-alone benchmark executables (one `add_executable` each in `CMakeLists.txt`), all built by `make all`. Not part of the library. |

Every header has a Doxygen `@file`/`@brief` block; keep it when adding files. `CMakeLists.txt` defines the INTERFACE target `pitts` (lists all headers), the optional `pitts_impl` object library, the benchmark executables, `pitts_py` (pybind11) and `pitts_jl` (JlCxx).

Scalar types supported everywhere: `float`, `double`, `std::complex<float>`, `std::complex<double>`.

Design goals and API/kernel conventions: see `DESIGN.md` (repository root).

## Coding style

There is no `.clang-format` on purpose (existing code would change massively). Follow the style of the surrounding code; the dominant conventions are:

- 2-space indentation, no tabs; Allman braces (`{` on its own line); namespace contents indented; `public:`/`private:` indented two spaces relative to `class`, members two more.
- `if( cond )` and `while( cond )` with spaces inside the parentheses; `for(int i = 0; i < n; i++)` without. No space before `(`.
- Compact arithmetic in index expressions: `i+1`, `r1*n*r2`, `offset+i`; spaces around comparison and logical operators (`i+1 < n && ...`).
- `template<typename T>` (no space), `T* p`, `const T& x`; `const auto` for local sizes and views.
- Types `int` for dimensions/ranks/loop indices over them, `long long` for row counts and chunk indices.
- Names: types `CamelCase`, functions/variables `camelCase` (`leftNormalize`, `nChunks`), internal kernels may use `snake_case` with a prefix (`normalize_contract1`, `t3_scale`), members with trailing underscore (`subTensors_`), template type parameter `T`.
- One-line getters as `{return x_;}`; single-statement loop/if bodies without braces are fine.
- No line-length limit, but wrap long parameter lists one parameter per line (see `solveMALS`).
- `#pragma omp ...` and preprocessor directives in column 0, even inside functions.
- Includes: standard headers first, then `pitts_*.hpp`; in `_impl.hpp` include the own `.hpp` first.
- Doxygen: `//!` line comments; each file starts with the SPDX header and a `/*! @file ... @brief ... */` block; every public function has `@param`/`@return`.
- Commented-out code blocks exist on purpose (debugging aids for numerical details); leave them alone unless asked.
- Error handling: `throw std::invalid_argument` for wrong arguments at the API boundary, `assert` for internal invariants.
- **Check existing functionality before writing helpers.** Before adding a small function (norm, axpy, scaling, reshape,
  contraction, concatenation, ...), look for it in the headers of the data structure you work with: for `Tensor3` read
  `pitts_tensor3.hpp` and all `pitts_tensor3_*.hpp` (not the `_impl.hpp` files, the declarations and Doxygen comments in
  the `.hpp` are enough), and likewise for `Tensor2`, `MultiVector` and `TensorTrain`; kernels used by several algorithms
  often live in `internal` of another header (`t3_nrm` in `pitts_tensortrain_norm.hpp`, `t3_axpy` in
  `pitts_tensortrain_axpby_normalized.hpp`, `t3_scale`/`normalize_contract1/2` in `pitts_tensortrain_normalize.hpp`,
  `concatLeftRight`/`concatTopBottom` in `pitts_tensor2_concat.hpp`). The tables below list what each header provides.
  Reuse is preferred over a private one-liner: no duplicated kernel, and the existing ones carry timers, OpenMP and
  dimension checks. A boundary case (first/last dimension) is usually handled with a 1x1 identity instead of a special case.

## Core data structures

| File | Content |
|---|---|
| `pitts_chunk.hpp` | `Chunk<T>`: fixed-size aligned SIMD block (the basic storage unit of all tensors). |
| `pitts_chunk_ops.hpp` | Dispatch header for chunk operations (fma, mul, bcast, ...) selecting the best implementation. |
| `pitts_chunk_ops_plain.hpp`, `pitts_chunk_ops_avx2.hpp`, `pitts_chunk_ops_avx512.hpp` | Scalar / AVX2 / AVX-512 implementations of the chunk operations. |
| `pitts_tensor2.hpp`, `_impl.hpp` | `Tensor2<T>`: small dense matrix (chunked, column-major). Also `ConstTensor2View`/`Tensor2View`. |
| `pitts_tensor3.hpp`, `_impl.hpp` | `Tensor3<T>`: rank-3 tensor (r1 x n x r2), the TT core type. |
| `pitts_multivector.hpp`, `_impl.hpp` | `MultiVector<T>`: tall-skinny matrix (many rows, few columns), chunked and padded. |
| `pitts_tensortrain.hpp`, `_impl.hpp` | `TensorTrain<T>`: list of `Tensor3` cores + dimensions, TT ranks, orthogonality flag; `copy`, `setZero/setOnes/setUnit`, sub-tensor access. |
| `pitts_tensortrain_operator.hpp`, `_impl.hpp` | `TensorTrainOperator<T>`: linear operator in TT format (row/column dims per core), `setEye`, sub-tensor access. |
| `pitts_fixed_tensor3.hpp`, `pitts_fixed_tensor3_apply.hpp`, `pitts_fixed_tensor3_combine.hpp`, `pitts_fixed_tensor3_split.hpp`, `pitts_fixed_tensor3_random.hpp` | Older variant with compile-time mode dimension `N` (`FixedTensor3<T,N>`) and its apply/combine/split/random operations. Used by the qubit simulator, not by the main algorithms. |
| `pitts_fixed_tensortrain.hpp`, `pitts_fixed_tensortrain_axpby.hpp`, `pitts_fixed_tensortrain_dot.hpp`, `pitts_fixed_tensortrain_random.hpp` | `FixedTensorTrain<T,N>` with axpby/dot/random operations (same remark). |
| `pitts_tensor2_eigen_adaptor.hpp`, `pitts_multivector_eigen_adaptor.hpp` | `EigenMap`/`ConstEigenMap` helpers giving an `Eigen::Map` onto Tensor2 / MultiVector data. |
| `pitts_eigen.hpp` | Includes Eigen with the project's defines and warning suppressions. |
| `pitts_mkl.hpp` | Optional Intel MKL include plus workarounds for non-Intel CPUs (`PITTS_DIRECT_MKL_GEMM`). |
| `pitts_missing_cereal.hpp` | Forward declarations so headers can offer cereal serialization without including cereal everywhere. |

## Operations on Tensor2 / Tensor3

| File | Content |
|---|---|
| `pitts_tensor2_random.hpp` | `randomize(Tensor2&)`. |
| `pitts_tensor2_concat.hpp` | Concatenate two matrices into a block matrix. |
| `pitts_tensor2_qb_decomposition.hpp` | QB part of the SVQB orthogonalization (Stathopoulos/Wu), used for rank-revealing normalization. |
| `pitts_tensor2_maxvol.hpp` | `internal::maxvol`: quasi-maximum-volume row selection of a tall matrix (Goreinov et al.), building block for TT-cross; optionally oversampled (more rows than columns, greedy volume growth, least-squares coefficients). |
| `pitts_tensor3_random.hpp` | `randomize(Tensor3&)`. |
| `pitts_tensor3_combine.hpp` | Contract two cores along r2/r1 into one larger core (MALS merging). |
| `pitts_tensor3_split.hpp` | Split a core into two by QR/SVD with rank truncation (`normalize_qb`, `normalize_svd`, ...); central truncation routine. |
| `pitts_tensor3_fold.hpp`, `pitts_tensor3_unfold.hpp` | Reshape Tensor2/MultiVector <-> Tensor3 (left/right unfoldings). |

## Operations on MultiVector (tall-skinny matrices)

| File | Content |
|---|---|
| `pitts_multivector_random.hpp` | `randomize(MultiVector&)` (thread-count independent). |
| `pitts_multivector_axpby.hpp` | Column-wise `axpy` (`y = alpha*x + y`), plus fused `axpy_dot` and `axpy_norm2`. |
| `pitts_multivector_dot.hpp` | Column-wise dot products. |
| `pitts_multivector_norm.hpp` | Column-wise 2-norms. |
| `pitts_multivector_scale.hpp` | Scale each column. |
| `pitts_multivector_gramian.hpp` | `X^T X`. |
| `pitts_multivector_cdist.hpp` | Pairwise squared distances between columns of two multivectors (k-means helper). |
| `pitts_multivector_centroids.hpp` | Weighted column sums / centroids (k-means helper). |
| `pitts_multivector_transform.hpp` | `Y = X * M` for small `M` (TSMM), optionally with reshape of the result. |
| `pitts_multivector_transpose.hpp` | Reshape + transpose of a tall-skinny matrix (used by TT-SVD). |
| `pitts_multivector_reshape.hpp` | Reshape keeping data order. |
| `pitts_multivector_triangular_mult.hpp` | In-place `X = X * R` with small upper-triangular `R`. |
| `pitts_multivector_triangular_solve.hpp` | In-place `X = X * R^-1` (backward substitution), optionally with column permutation. |
| `pitts_multivector_tsqr.hpp` | `block_TSQR`: MPI+OpenMP Householder tall-skinny QR returning only R (Q-less). Core kernel of the fast TT-SVD. |

## Operations on TensorTrain

| File | Content |
|---|---|
| `pitts_tensortrain_random.hpp` | `randomize(TensorTrain&)`. |
| `pitts_tensortrain_norm.hpp` | `norm2` (exploits orthogonality). |
| `pitts_tensortrain_dot.hpp` | Inner product of two TTs. |
| `pitts_tensortrain_normalize.hpp` | `leftNormalize`, `rightNormalize`, `normalize` with rank truncation (uses Q-less TSQR unless `PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB`). |
| `pitts_tensortrain_axpby.hpp` | `axpby`: dispatches to the normalized or plain variant. |
| `pitts_tensortrain_axpby_plain.hpp` | Classical TT addition (concatenate cores, then truncate). |
| `pitts_tensortrain_axpby_normalized.hpp` | Faster addition when one operand is orthogonalized (optional `PITTS_TENSORTRAIN_AXPBY_REORTHOGONALIZE`). |
| `pitts_tensortrain_from_dense.hpp` | Fast TT-SVD of a dense tensor given as MultiVector, TSQR-based (main algorithm, incl. "thick bounds"/auto-fusing variants). |
| `pitts_tensortrain_from_dense_classical.hpp` | Classical TT-SVD (reference, Eigen SVD). |
| `pitts_tensortrain_from_dense_twosided.hpp` | Two-sided TSQR variant of the fast TT-SVD. |
| `pitts_tensortrain_to_dense.hpp` | Expand a TT to a dense MultiVector / array. |
| `pitts_tensortrain_gram_schmidt.hpp` | Modified/iterated Gram-Schmidt for TT vectors (used by TT-GMRES). |
| `pitts_tensortrain_cross.hpp`, `_impl.hpp` | `applyElementwise`: y ~ f(x) or y ~ f(x1, x2) (e.g. element-wise product) entry-wise via TT-cross (ALS-cross with AMEn-style enrichment in the left-to-right sweeps, or DMRG-cross with `nMALS = 2`; y in the orthogonal gauge as in `solveMALS`, so the local truncation is a truncation of y; optional oversampling = least-squares interpolation on more rows/columns than the rank), options/info structs, scalar or array functor (see `applyToBlock`), algorithm description in the file header. The sweep driver (`_impl.hpp`, `internal::tt_cross::crossSweeps`, written for a function of N tensor trains) is templated on the functor and therefore always included (no explicit instantiation). |
| `pitts_tensortrain_cross_helper.hpp`, `_impl.hpp` | Building blocks for TT-cross: nested index sets (decoding from unfoldings, expansion, random sets), partial contractions `L_{k-1} X_k` / `X_k R_{k+1}` as `SweepData` lambdas with the interface matrices as row/column selections (`interfaceLeft/Right`, boundary as `std::nullopt`), local block assembly (`combineCores`, `localBlockFromLeft/Right`), `applyToBlock` (scalar functor per entry or array functor on the whole block as a MultiVector; further blocks become further function arguments), `t3_rms`/`rmsDifference` (size-independent convergence measures), `randomIndices`/`evaluateAt` (independent control sample), `FunctionOnCrosses` (everything of the driver that depends on the N input tensors: their cached partial contractions on the crosses of y and of the error approximation, block assembly and the call of f), `solveLeftInterface/solveRightInterface` (inverse of the interface contractions: samples on a cross into the coordinates of y, least squares with oversampling), `maxvolColumns`, `enrichFactors` (append new directions, with a Gram-matrix-corrected projection because the orthogonal factor is only reconstructed), local core updates of y in the orthogonal gauge (`updateCoreLeft/Right`: truncated SVD, optional enrichment, orthonormal core, maxvol on the values of the new left/right part of y, the interface is the 1x1 identity at the boundary; with oversampling more rows/columns than the rank), `dominantRows/Columns` (index selection for the error crosses, which have no cores), `initialRightFrame` (zero y with the right-orthonormal cores of the orthogonalized direct sum of the inputs as frame, optionally truncated), `rightIndexSetsFromFrame` (maxvol sweep over the right-orthonormal cores of a frame: initial sets of y and of the error crosses). |
| `pitts_tensortrain_laplace_operator.hpp` | Apply a simple Laplace stencil to a TT (test/bench helper). |
| `pitts_tensortrain_debug.hpp` | Operator overloads (`+`, `-`, `*`) on TTs for tests, not for production use. |
| `pitts_tensortrain_from_itensor.hpp`, `pitts_tensortrain_to_itensor.hpp` | Conversion from/to ITensor MPS (needs ITensor). |

## Operations on TensorTrainOperator

| File | Content |
|---|---|
| `pitts_tensortrain_operator_apply.hpp` | `apply(TTOp, TT, TT)`: operator times TT vector. |
| `pitts_tensortrain_operator_apply_transposed.hpp` | Transposed operator times TT vector. |
| `pitts_tensortrain_operator_apply_op.hpp`, `pitts_tensortrain_operator_apply_transposed_op.hpp` | Operator times operator (product of two TT operators), plain and transposed. |
| `pitts_tensortrain_operator_apply_dense.hpp`, `pitts_tensortrain_operator_apply_transposed_dense.hpp` | Operator (or its transpose) times dense MultiVector (used inside MALS; padded contraction unless `PITTS_TENSORTRAIN_MALS_SLOWCONTRACT`). |
| `pitts_tensortrain_operator_to_dense.hpp` | Expand a TT operator to a dense matrix. |
| `pitts_tensortrain_operator_to_qtt.hpp` | Split every dimension into 2x2 factors (QTT format). |
| `pitts_tensortrain_operator_debug.hpp` | Operator overloads for tests. |
| `pitts_tensortrain_operator_from_itensor.hpp`, `pitts_tensortrain_operator_to_itensor.hpp` | Conversion from/to ITensor MPO. |
| `pitts_tensortrain_operator_itensor_autompo_pybind.*` | Python binding wrapping ITensor AutoMPO (only with ITensor). |

## Solvers

| File | Content |
|---|---|
| `pitts_gmres.hpp` | Generic GMRES templated on vector/operator type (used for the dense local problems in MALS). |
| `pitts_tensortrain_solve_gmres.hpp` | `solveGMRES`: TT-GMRES with rank truncation. |
| `pitts_tensortrain_solve_mals.hpp` | `solveMALS`: ALS / MALS / AMEn family (symmetric and non-symmetric via Petrov-Galerkin), many options. |
| `pitts_tensortrain_solve_mals_helper.hpp` | Building blocks for MALS: left/right contractions, local operator/RHS, AMEn residual enrichment. |
| `pitts_tensortrain_solve_mals_debug.hpp` | Expensive consistency checks for MALS (only in debug/tests). |
| `pitts_tensortrain_sweep_index.hpp` | `SweepIndex`: iterates over core positions for sweeps (ALS width 1, MALS width 2, ...). |
| `pitts_tensortrain_sweep_data.hpp` | `SweepData`/`defineSweepData`: cache of per-dimension results computed incrementally from the left and right with invalidation (partial contractions in MALS and TT-cross). |
| `pitts_ttn_local_gradient_contraction.hpp` | Special contraction for local gradients in tree tensor networks (used from Python). |

## Infrastructure / helpers

| File | Content |
|---|---|
| `pitts_common.hpp` | `initialize` (MPI_Init if needed, prints thread/process info, sets random seed) and `finalize`. |
| `pitts_parallel.hpp` | `mpiProcInfo`, `ompThreadInfo`, `distribute` (row partitioning), `mpiGather`, `mpiCombineMaps`, MPI type mapping. |
| `pitts_random.hpp` | Seeded random generator and distributions (per-thread). |
| `pitts_timer.hpp` | Scoped timers, `TimingStatistics`, `printTimingStatistics`. |
| `pitts_performance.hpp` | Scoped timers with flop/byte counts (`createScopedTimer`), `printPerformanceStatistics`, likwid marker API integration. |
| `pitts_kernel_info.hpp` | Types for flop/byte accounting (`Flops`, `Bytes`, `kernel_info::FMA<T>()` ...). |
| `pitts_machine_info.hpp` | Cache sizes (likwid or hard-coded fallback). |
| `pitts_scope_info.hpp` | Compile-time function/type/argument info for timers. |
| `pitts_hash_function.hpp` | Compile-time string hash (`djb_hash`). |
| `pitts_type_name.hpp` | Type name as string. |
| `pitts_qubit_simulator.hpp` | `QubitSimulator`: simplistic gate-based quantum circuit simulator built on `FixedTensorTrain` (the main user of the fixed-dimension types). |

## Bindings

**Release the GIL for everything that takes noticeable time** (`py::call_guard<py::gil_scoped_release>()` as
the last argument of `m.def`): a C++ call holding the GIL blocks all python threads, and in jupyter that
deadlocks the kernel as soon as the C++ code writes more than the pipe buffer (64 KiB) to `std::cout`
(ipykernel captures the file descriptors and drains them in a python thread, which needs the GIL).
Only for functions whose body does not touch python objects - check the wrapper, not the `m.def` line:
`gramSchmidt` (a `py::list`), the `getSubTensor`/`setSubTensor`/`fromDense_classical`/`toDense` helpers and
`cdist2`/`block_TSQR`/`ttn_local_gradient_contraction` (they create numpy arrays) release the GIL inside,
around the calculation only. A python callback needs `py::gil_scoped_acquire` (see `applyElementwise`).

| File | Content |
|---|---|
| `pitts_pybind.cpp` | Defines module `pitts_py`; pulls in `*_pybind.cpp` for QubitSimulator, TensorTrain, TensorTrainOperator, solvers, MultiVector, common. Python class names carry the type suffix, e.g. `TensorTrain_double`, `MultiVector_float`. |
| `pitts_common_pybind.*` | `initialize`, `finalize`, performance/timing statistics. |
| `pitts_multivector_pybind.*` | MultiVector with numpy buffer protocol, `randomize`, `cdist2`, ... |
| `pitts_tensortrain_pybind.*` | TensorTrain: from/to dense, axpby, dot, norm, normalize, gramSchmidt, sub-tensors. |
| `pitts_tensortrain_operator_pybind.*` | TensorTrainOperator: apply, applyT, axpby, toDense, toQtt, ... |
| `pitts_tensortrain_solve_pybind.*` | `solveGMRES`, `solveMALS` and their option enums. |
| `pitts_tensortrain_cross_pybind.*` | `applyElementwise` (TT-cross) with one input `(f, TTx, TTy)` or two inputs `(f, TTx1, TTx2, TTy)`: the python function gets 1d numpy arrays viewing a block of tensor entries (one per input) and returns an array of the same size (or modifies its first argument in-place and returns None); the TT-cross options are keyword arguments (defined once for both variants). Releases the GIL around the algorithm. |
| `pitts_qubit_simulator_pybind.*` | QubitSimulator. |
| `pitts_jlbind.cpp`, `*_jlbind.*`, `PittsJl.jl` | Julia binding; currently only QubitSimulator, Tensor2, MultiVector and common are enabled (TensorTrain parts commented out). |

## Benchmark executables (`*_bench.cpp`)

All call `PITTS::initialize`, take sizes from `argv`, loop `nIter` times and print timing/performance statistics. Grouped by topic:

- **Dense kernels**: `gemm_bench`, `tsmm_bench`, `tsmm_inplace_bench`, `tsmm_single_bench` (MultiVector x small matrix), `trmm_bench`, `trsm_bench` (triangular mult/solve), `transpose_bench`, `multivector_copy_bench`, `multivector_axpy_bench`, `qr_bench`, `svd_bench` (Eigen/LAPACK QR/SVD on Tensor2), `normalize_qb_bench` (Tensor3 split), `tsqr_bench.cpp` (compiled four times as `tsqr_{double,single,complex_double,complex_single}_bench`), `tsqr_transform_block_calc_bench` (inner TSQR kernel variants), `tsqr_parallel.cpp` (experiment: thread synchronization scheme with `std::latch` for a parallel TSQR, dummy computations).
- **TT-SVD**: `tt_from_dense_bench`, `tt_from_dense_classical_bench`, `tt_from_dense_twosided_bench`, `tt_from_dense_thickbounds_bench`, `tt_from_dense_thickbounds_single_bench`, `tt_from_dense_autofusing_bench`, `tt_from_dense_twosided_autofusing_bench`, `tt_from_dense_twosided_thickbounds_bench`; `unary_truncation_bench`, `binary_truncation_bench`, `binary_orthogonalization_bench` (TSQR-based truncation/orthogonalization building blocks).
- **TT operations**: `axpby_bench`, `axpby_left_ortho_bench`, `axpby_right_ortho_bench`, `dot_bench`, `norm2_bench`, `normalize_bench`, `apply_op_bench`, `apply_op_dense_bench`, `cg_bench` (CG with Laplace operator), `tt_cross_bench` (TT-cross round trip: element-wise square of a random TT, then element-wise sqrt, timings per TT-cross and the real error of the result).
- **Other**: `cdist2_bench`, `centroids_bench`, `kmeans_bench` (k-means on MultiVectors), `qubit_simulator_bench`, `ttn_local_gradient_bench`.
