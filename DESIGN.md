<!--
SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
SPDX-License-Identifier: BSD-3-Clause
-->

# PITTS design guidelines

These guidelines describe how the library is meant to be structured and extended.
They are derived from the existing code and are not strict rules: apply them where they
matter (numerical kernels, data structures, public API) and prefer readability elsewhere
(argument parsing, small `std::vector`/`std::string` handling, test code, examples).

## Goals

1. **High code quality with a small, well-designed interface.** PITTS aims to combine
   high performance with template generality (one implementation for all scalar types,
   composable free functions on a few data types) without sacrificing readability of the
   interface. Every public function should be documented, tested and easy to call
   correctly; complexity belongs into the `_impl.hpp` files, not into the API. Prefer a
   clear design over a clever one, and keep the interface stable and minimal.
2. **Performance on multi-core CPUs (and clusters).** The core kernels are written for
   SIMD (`Chunk<T>`, aligned and padded storage), OpenMP parallelism with explicit thread
   distribution, and memory reuse. Performance-critical code may be less pretty than
   library-style C++; that is a deliberate trade-off, but only where it pays off.
3. **Measurability.** Every kernel reports runtime, flops and data transfers through
   `pitts_performance.hpp`, so the roofline position of each operation can be checked.
4. **Small, composable operations.** One operation per header, free functions on a few
   data types (`Tensor2`, `Tensor3`, `MultiVector`, `TensorTrain`, `TensorTrainOperator`).
   Higher-level algorithms (normalize, axpby, TT-SVD, GMRES, MALS/AMEn) are built from these.
5. **Complete documentation.** Every public function, class and file has Doxygen
   documentation (`@brief`, `@param`, `@return`); only internal helpers may be terse.
   REUSE/SPDX compliance is checked in CI.
6. **Reproducible correctness.** Every operation has unit tests for all supported scalar
   types (`float`, `double`, `std::complex<float>`, `std::complex<double>`), usually
   against a dense/Eigen reference.

## Structure rules

- **New operation = file triple + test (+ binding).** `pitts_<type>_<op>.hpp`
  (interface + docs), `pitts_<type>_<op>_impl.hpp` (implementation),
  `pitts_<type>_<op>.cpp` (explicit instantiations for `PITTS_DEVELOP_BUILD`),
  `test/test_<type>_<op>.cpp`; add to `src/CMakeLists.txt` (`pitts` target and
  `pitts_impl` list). Expose it in `pitts_py` if it is user-facing.
- **Keep includes consistent with and without `PITTS_DEVELOP_BUILD`.** The `.hpp` must be
  self-contained; the `_impl.hpp` includes what it needs itself. CI builds both variants.
- **Algorithm variants** for benchmarking are selected with compile definitions
  (`PITTS_TENSORTRAIN_*`, `PITTS_DIRECT_MKL_GEMM`), not with runtime flags, so that the
  production path has no overhead.

## API conventions

- Prefer **free functions with output parameters** (`void op(const A& a, B& result)`) to
  returning large objects, so callers can reuse allocated memory. Small results
  (scalars, `std::vector<int>` of ranks) are returned by value. Free functions on PITTS
  types are found by argument-dependent lookup, so calling code (benchmarks, examples,
  user code) can write `copy(a, b)`, `normalize(TT)`, `dot(x, y)` without the `PITTS::`
  prefix; keep new operations in namespace `PITTS` (or `PITTS::internal`) for this to work.
- **Explicit copies:** `copy(a, b)`; copy assignment of the large types is deleted, moves
  are allowed and cheap.
- **Argument validation** at the API boundary with exceptions (`std::invalid_argument` for
  dimension mismatches, `std::out_of_range` for indices); **internal invariants** with
  `assert`.
- Console output only behind a `verbose`-style parameter or in `initialize`/statistics
  printing. (Some `std::cout` calls in `_impl` files are work-in-progress leftovers.)
- Sizes: `int` for TT dimensions and ranks, `long long` for row counts and chunk indices
  of `MultiVector`.

## Kernel conventions (performance-critical code only)

- Wrap the body in `const auto timer = PITTS::performance::createScopedTimer<...>(...)`
  with arguments, a flop count and a data-transfer count (`kernel_info::*`). Use
  `timing::createScopedTimer` for non-kernel functions.
- Avoid allocations inside loops; resize output objects once at the start.
- Parallelize over chunks with `#pragma omp parallel` and
  `internal::parallel::distribute(...)`; keep the serial fallback correct for
  `OMP_NUM_THREADS=1`.
- Dense linear algebra: there are usually two variants, a simple Eigen implementation
  (easy to read and to write, used by default) and an optional low-level BLAS/LAPACK call
  through Intel MKL (`PITTS_DIRECT_MKL_GEMM`, `PITTS_EIGEN_USE_LAPACKE`). Start with the
  Eigen variant; add the direct call only when measurements show a benefit.
- MPI: most operations are local. MPI communication is confined to a few documented
  global operations (TSQR, reductions in dot/norm, gathering in `to_dense`); use
  `internal::parallel::*` helpers rather than raw MPI calls where possible.

## Numerical soundness

Follow the best practices of numerical linear algebra, and check the soundness of an algorithm step by step:

- No explicit inverses: solve with a factorization (QR, Cholesky, LU with pivoting, SVD).
- No normal equations or Gram matrices where an orthogonal factorization works (they square the
  condition number).
- No avoidable cancellation; relative (scale-invariant) tolerances; no decisions on round-off.
- Consider the conditioning and the stability of every step (which quantities can become
  ill-conditioned, what bounds the error), and document the argument in the code where it is not obvious.

## TensorTrain specifics

- `TensorTrain` tracks the orthogonality state of each core (`TT_Orthogonality`).
  Every function that replaces cores must pass the correct state to `setSubTensor(s)`
  so that later operations can skip normalization sweeps.
- Cores are exchanged by moving (`setSubTensor` returns the old core) to reuse memory.
- Boundary ranks `r0`/`rd` may differ from 1 in some algorithms; do not assume `r0 == 1`.

## Testing

- Compare against a dense reference (`toDense`, Eigen). The typical tolerance is
  `std::sqrt(std::numeric_limits<T>::epsilon())` (i.e. ~1e-8 for double, ~3e-4 for float):
  strict enough to catch real errors, loose enough for floating-point deviations from
  different summation orders or thread counts. Cover rank-deficient, rank-1 and boundary
  cases.
- Tests must pass with 1 and 3 MPI processes and with 4 OpenMP threads; tests that need
  a specific thread/process count check for it explicitly.
