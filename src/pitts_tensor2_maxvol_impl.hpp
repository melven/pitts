// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor2_maxvol_impl.hpp
* @brief Quasi-maximum-volume row selection (maxvol) for tall matrices, used by TT-cross algorithms
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-11
*
**/

// include guard
#ifndef PITTS_TENSOR2_MAXVOL_IMPL_HPP
#define PITTS_TENSOR2_MAXVOL_IMPL_HPP

// includes
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "pitts_tensor2_maxvol.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_eigen.hpp"
#include "pitts_kernel_info.hpp"
#include "pitts_performance.hpp"
#include "pitts_timer.hpp"
#ifdef PITTS_DIRECT_MKL_GEMM
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
#ifdef PITTS_DIRECT_MKL_GEMM
    inline lapack_int maxvol_getrf_mapper(lapack_int m, lapack_int n, double* A, lapack_int lda, lapack_int* ipiv)
    {
      return LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, A, lda, ipiv);
    }

    inline lapack_int maxvol_getrf_mapper(lapack_int m, lapack_int n, float* A, lapack_int lda, lapack_int* ipiv)
    {
      return LAPACKE_sgetrf(LAPACK_COL_MAJOR, m, n, A, lda, ipiv);
    }

    inline lapack_int maxvol_getrf_mapper(lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda, lapack_int* ipiv)
    {
      return LAPACKE_zgetrf(LAPACK_COL_MAJOR, m, n, (MKL_Complex16*)A, lda, ipiv);
    }

    inline lapack_int maxvol_getrf_mapper(lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda, lapack_int* ipiv)
    {
      return LAPACKE_cgetrf(LAPACK_COL_MAJOR, m, n, (MKL_Complex8*)A, lda, ipiv);
    }

    inline void maxvol_trsm_mapper(const CBLAS_LAYOUT Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE transa, const CBLAS_DIAG diag, const CBLAS_INDEX M, const CBLAS_INDEX N, const double alpha, const double * A, const CBLAS_INDEX lda, double * B, const CBLAS_INDEX ldb)
    {
      cblas_dtrsm(Layout, side, uplo, transa, diag, M, N, alpha, A, lda, B, ldb);
    }

    inline void maxvol_trsm_mapper(const CBLAS_LAYOUT Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE transa, const CBLAS_DIAG diag, const CBLAS_INDEX M, const CBLAS_INDEX N, const float alpha, const float * A, const CBLAS_INDEX lda, float * B, const CBLAS_INDEX ldb)
    {
      cblas_strsm(Layout, side, uplo, transa, diag, M, N, alpha, A, lda, B, ldb);
    }

    inline void maxvol_trsm_mapper(const CBLAS_LAYOUT Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE transa, const CBLAS_DIAG diag, const CBLAS_INDEX M, const CBLAS_INDEX N, const std::complex<double> alpha, const std::complex<double> * A, const CBLAS_INDEX lda, std::complex<double> * B, const CBLAS_INDEX ldb)
    {
      cblas_ztrsm(Layout, side, uplo, transa, diag, M, N, (const MKL_Complex16*)&alpha, (const MKL_Complex16*)A, lda, (MKL_Complex16*)B, ldb);
    }

    inline void maxvol_trsm_mapper(const CBLAS_LAYOUT Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE transa, const CBLAS_DIAG diag, const CBLAS_INDEX M, const CBLAS_INDEX N, const std::complex<float> alpha, const std::complex<float> * A, const CBLAS_INDEX lda, std::complex<float> * B, const CBLAS_INDEX ldb)
    {
      cblas_ctrsm(Layout, side, uplo, transa, diag, M, N, (const MKL_Complex8*)&alpha, (const MKL_Complex8*)A, lda, (MKL_Complex8*)B, ldb);
    }

    //! LU decomposition with partial pivoting P A = L U of the tall matrix A for the initial row selection (just for timings)
    //!
    //! LU contains L (unit lower trapezoidal, m x r) and U (upper triangular, r x r), row k of P A is row perm[k] of A.
    //!
    template<typename T>
    void maxvol_lu(const ConstTensor2View<T>& A, Tensor2<T>& LU, std::vector<int>& perm)
    {
      const auto m = A.r1();
      const auto r = A.r2();
      // flops of ?getrf for a tall matrix
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r"},{m, r}}, // arguments
        {{(m*r*r/2. - r*r*r/6.)*kernel_info::FMA<T>()}, // flops
         {(m*r)*kernel_info::Load<T>() + (m*r)*kernel_info::Store<T>()}} // data transfers
        );

      LU.resize(m, r);
      auto mapLU = EigenMap(LU);
      mapLU = ConstEigenMap(A);
      std::vector<lapack_int> ipiv(r);
      // info > 0 only reports an exactly zero pivot, rank deficiency is checked by the caller
      const lapack_int info = maxvol_getrf_mapper(m, r, mapLU.data(), mapLU.colStride(), ipiv.data());
      if( info < 0 )
        throw std::runtime_error("maxvol: ?getrf failed with info " + std::to_string(info) + "!");

      // the row interchanges of ?getrf (1-based, applied one after another) as a permutation
      perm.resize(m);
      for(int i = 0; i < m; i++)
        perm[i] = i;
      for(int k = 0; k < r; k++)
        std::swap(perm[k], perm[ipiv[k]-1]);
    }

    //! calculate B = A * A(I,:)^{-1} = P^T [I; L2 L1^{-1}] from the LU decomposition P A = [L1; L2] U (just for timings)
    template<typename T>
    void maxvol_lu_solve(Tensor2<T>& LU, const std::vector<int>& perm, Tensor2<T>& B)
    {
      const auto m = LU.r1();
      const auto r = LU.r2();
      // triangular solve with m-r right-hand sides
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r"},{m, r}}, // arguments
        {{((m-r)*r*r/2.)*kernel_info::FMA<T>()}, // flops
         {(m*r)*kernel_info::Load<T>() + (m*r)*kernel_info::Store<T>()}} // data transfers
        );

      // L2 <- L2 L1^{-1} in place
      auto mapLU = EigenMap(LU);
      maxvol_trsm_mapper(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, m-r, r, T(1), mapLU.data(), mapLU.colStride(), mapLU.data()+r, mapLU.colStride());

      B.resize(m, r);
      auto mapB = EigenMap(B);
      for(int k = 0; k < r; k++)
      {
        mapB.row(perm[k]).setZero();
        mapB(perm[k], k) = T(1);
      }
      for(int k = r; k < m; k++)
        mapB.row(perm[k]) = mapLU.row(k);
    }
#else
    //! small wrapper around the column-pivoted QR of A^H used for the initial row selection (just for timings)
    template<typename T>
    auto maxvol_qr(const ConstTensor2View<T>& A)
    {
      const auto m = A.r1();
      const auto r = A.r2();
      // QR of the r x m matrix A^H, flops as reported by LAPACKE for a full-rank QR
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r"},{m, r}}, // arguments
        {{(r*r*(3.*m-r)/3)*kernel_info::FMA<T>()}, // flops
         {(m*r)*kernel_info::Load<T>() + (m*r)*kernel_info::Store<T>()}} // data transfers
        );

      using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      return Eigen::ColPivHouseholderQR<mat>(ConstEigenMap(A).adjoint());
    }

    //! calculate B = A * A(I,:)^{-1} with an LU decomposition of the selected rows (just for timings)
    template<typename T>
    void maxvol_solve(const ConstTensor2View<T>& A, const std::vector<int>& I, Tensor2<T>& B)
    {
      const auto m = A.r1();
      const auto r = A.r2();
      // LU decomposition (r^3/3) plus two triangular solves with m right-hand sides (m r^2)
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r"},{m, r}}, // arguments
        {{(1.*m*r*r + r*r*r/3.)*kernel_info::FMA<T>()}, // flops
         {(m*r)*kernel_info::Load<T>() + (m*r)*kernel_info::Store<T>()}} // data transfers
        );

      using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      const auto mapA = ConstEigenMap(A);

      mat Ahat(r, r);
      for(int j = 0; j < r; j++)
        Ahat.row(j) = mapA.row(I[j]);

      // solve Ahat^H B^H = A^H for B^H
      const Eigen::PartialPivLU<mat> luH(Ahat.adjoint());
      B.resize(m, r);
      EigenMap(B) = luH.solve(mapA.adjoint()).adjoint();
    }
#endif

    //! entry with the largest |B_ij|^2 for maxvol_swapRows
    template<typename RealType>
    struct MaxvolMaxEntry
    {
      RealType abs2;
      long long i, j;

      //! the larger one; ties: the first one in column-major order (as Eigen's maxCoeff), so the result does not depend on the order of the comparisons
      static MaxvolMaxEntry larger(const MaxvolMaxEntry& a, const MaxvolMaxEntry& b)
      {
        return (b.abs2 > a.abs2 || (b.abs2 == a.abs2 && (b.j < a.j || (b.j == a.j && b.i < a.i)))) ? b : a;
      }
    };

// OpenMP reduction for the largest entry, at namespace scope for each type (a declaration in a function template crashes some GCC versions)
#pragma omp declare reduction(maxvolMaxEntry : MaxvolMaxEntry<float> : omp_out = MaxvolMaxEntry<float>::larger(omp_out, omp_in)) initializer(omp_priv = MaxvolMaxEntry<float>{-1.f, 0, 0})
#pragma omp declare reduction(maxvolMaxEntry : MaxvolMaxEntry<double> : omp_out = MaxvolMaxEntry<double>::larger(omp_out, omp_in)) initializer(omp_priv = MaxvolMaxEntry<double>{-1., 0, 0})

    //! swap rows into the selection while max |B| > threshold, at most maxIter times (just for timings)
    //!
    //! B = A * A(I,:)^{-1}; when row i replaces the j-th selected row: B <- B - B(:,j) * (B(i,:) - e_j^T) / B(i,j),
    //! so B(I,:) stays the identity for the new selection. The update and the search for the next maximum are done in
    //! one pass over B, parallelized over the columns (one parallel loop per swap, the result does not depend on the
    //! number of threads).
    //!
    template<typename T, typename RealType>
    void maxvol_swapRows(Tensor2<T>& B, std::vector<int>& I, RealType threshold, int maxIter)
    {
      // the number of swaps is only known afterwards, so no flops/data transfers
      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      const long long m = B.r1();
      const long long r = B.r2();

      using MaxEntry = MaxvolMaxEntry<RealType>;
      // maximum of column k with its position (only for a column that may contain a new maximum, rarely); scanned again
      // instead of searching for the value of the vectorized loop, which may round differently (e.g. FMA for complex)
      const auto inColumn = [&B, m](long long k)
      {
        MaxEntry result{Eigen::numext::abs2(B(0,k)), 0, k};
        for(long long l = 1; l < m; l++)
        {
          const RealType v = Eigen::numext::abs2(B(l,k));
          if( v > result.abs2 )
            result = MaxEntry{v, l, k};
        }
        return result;
      };

      // workspace, reused for all swaps (B(:,j) and B(i,:) are overwritten by the update)
      Eigen::Matrix<T, Eigen::Dynamic, 1> Bj(m);
      Eigen::Matrix<T, 1, Eigen::Dynamic> Bi(r);

      // initial maximum (threads only for larger matrices, one fork/join per swap)
      MaxEntry best{RealType(-1), 0, 0};
#pragma omp parallel for schedule(static) reduction(maxvolMaxEntry : best) if(m*r > 100000)
      for(long long k = 0; k < r; k++)
      {
        RealType colMax = RealType(-1);
        for(long long l = 0; l < m; l++)
          colMax = std::max(colMax, Eigen::numext::abs2(B(l,k)));
        if( colMax > best.abs2 )
          best = MaxEntry::larger(best, inColumn(k));
      }

      const RealType threshold2 = threshold*threshold;
      for(int iter = 0; iter < maxIter; iter++)
      {
        if( best.abs2 <= threshold2 )
          break;

        // swap row i into the selection, replacing the j-th selected row
        const long long i = best.i, j = best.j;
        const T pivot = B(i,j);
        Bj = ConstEigenMap(B).col(j);
        Bi = ConstEigenMap(B).row(i);
        Bi(j) -= T(1);
        Bi /= pivot;
        I[j] = i;

        // rank-1 update B <- B - Bj * Bi and the next maximum in one pass
        best = MaxEntry{RealType(-1), 0, 0};
#pragma omp parallel for schedule(static) reduction(maxvolMaxEntry : best) if(m*r > 100000)
        for(long long k = 0; k < r; k++)
        {
          RealType colMax = RealType(-1);
          for(long long l = 0; l < m; l++)
          {
            B(l,k) -= Bj(l) * Bi(k);
            colMax = std::max(colMax, Eigen::numext::abs2(B(l,k)));
          }
          if( colMax > best.abs2 )
            best = MaxEntry::larger(best, inColumn(k));
        }
      }
    }

    //! add row i to an oversampled selection, with C = B M only the small matrix M is updated (just for timings)
    //!
    //! M <- M - beta (M c^H) c with c = C(i,:) = B(i,:) M, and the scores |C(j,:)|^2 are updated directly; B is only read.
    //! c (1 x r), w (r) and Cc (m) are workspace of the correct size, reused for all added rows.
    //!
    template<typename T, typename RealType>
    void maxvol_addRow(long long i, const Tensor2<T>& B, Tensor2<T>& M, Eigen::Matrix<RealType, Eigen::Dynamic, 1>& scores,
                       Eigen::Matrix<T, 1, Eigen::Dynamic>& c, Eigen::Matrix<T, Eigen::Dynamic, 1>& w, Eigen::Matrix<T, Eigen::Dynamic, 1>& Cc)
    {
      const auto m = B.r1();
      const auto r = B.r2();
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r"},{m, r}}, // arguments
        {{(1.*m*r + 3.*r*r)*kernel_info::FMA<T>()}, // flops
         {(m*r)*kernel_info::Load<T>() + m*kernel_info::Update<T>()}} // data transfers
        );

      // C = A L is replaced by A L' with L' = L (I - beta c^H c), where (I - beta c^H c)^2 = I - c^H c / (1 + |c|^2), so that
      // L' L'^H = (G + u^H u)^{-1} for the Gram matrix G of the selection and the new row u (Sherman-Morrison);
      // beta = (1 - 1/sqrt(1+a))/a = 1/(sqrt(1+a) (1+sqrt(1+a))) with a = |c|^2, the second form is stable for small a.
      // With C = B M: C' = B M (I - beta c^H c) = B (M - beta w c) with w = M c^H, and
      // |C'(j,:)|^2 = C(j,:) (I - c^H c / (1+a)) C(j,:)^H = |C(j,:)|^2 - |(C c^H)_j|^2 / (1+a) with C c^H = B w.
      const auto mapB = ConstEigenMap(B);
      auto mapM = EigenMap(M);
      c.noalias() = mapB.row(i) * mapM;
      const RealType a = c.squaredNorm();
      const RealType sqrt1a = std::sqrt(1 + a);
      const RealType beta = 1 / (sqrt1a * (1 + sqrt1a));
      w.noalias() = mapM * c.adjoint();
      Cc.noalias() = mapB * w;
      scores -= Cc.cwiseAbs2() / (1 + a);
      // scale w in place (no temporary for beta * w), it is not needed afterwards
      w *= beta;
      mapM.noalias() -= w * c;
    }

    //! interpolation coefficients C C(I,:)^H = A A(I,:)^+ of an oversampled selection with C = B M (just for timings)
    template<typename T>
    void maxvol_coefficients(const Tensor2<T>& B, const Tensor2<T>& M, const std::vector<int>& I, Tensor2<T>& AAinv)
    {
      const auto m = B.r1();
      const auto r = B.r2();
      const long long q = I.size();
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r", "q"},{m, r, q}}, // arguments
        {{(1.*m*r*q + 2.*r*r*q)*kernel_info::FMA<T>()}, // flops
         {(m*r)*kernel_info::Load<T>() + (m*q)*kernel_info::Store<T>()}} // data transfers
        );

      using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      const auto mapB = ConstEigenMap(B);
      const auto mapM = ConstEigenMap(M);
      // the selected rows as an explicit loop (an Eigen indexed view triggers -Warray-bounds false positives with GCC 13)
      mat BI(q, r);
      for(int k = 0; k < q; k++)
        BI.row(k) = mapB.row(I[k]);
      // C C(I,:)^H = B M (B(I,:) M)^H: first the small r x q factor, then one product with B
      const mat CI = BI * mapM;
      const mat K = mapM * CI.adjoint();
      AAinv.resize(m, q);
      EigenMap(AAinv).noalias() = mapB * K;
    }

    // implement maxvol
    template<typename T>
    std::vector<int> maxvol(const ConstTensor2View<T>& A, T tol, int maxIter, Tensor2<T>* AAinv, int nRows)
    {
      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      using RealType = decltype(std::abs(T()));
      using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

      const int m = A.r1();
      const int r = A.r2();
      if( r < 1 || m < r )
        throw std::invalid_argument("maxvol: requires a tall matrix (m x r) with m >= r >= 1!");
      if( nRows != 0 && nRows < r )
        throw std::invalid_argument("maxvol: requires at least as many rows to select as columns!");
      nRows = nRows == 0 ? r : std::min(nRows, m);

      const RealType threshold = 1 + std::real(tol);

      // initial selection: the pivot rows of an LU decomposition with partial pivoting of A (BLAS-3, classical maxvol
      // initialization) with MKL, otherwise of a column-pivoted QR decomposition of A^T (greedy choice of the "largest" rows)
      std::vector<int> I(r);
#ifdef PITTS_DIRECT_MKL_GEMM
      Tensor2<T> LU;
      std::vector<int> perm;
      maxvol_lu(A, LU, perm);
      for(int j = 0; j < r; j++)
        I[j] = perm[j];
      const Eigen::Matrix<RealType, Eigen::Dynamic, 1> absDiag = ConstEigenMap(LU).diagonal().array().abs();
#else
      const auto qr = maxvol_qr(A);
      const auto& pivots = qr.colsPermutation().indices();
      for(int j = 0; j < r; j++)
        I[j] = pivots(j);
      const Eigen::Matrix<RealType, Eigen::Dynamic, 1> absDiag = qr.matrixR().diagonal().array().abs();
#endif

      // check for a rank-deficient input (A(I,:) would be singular): count the diagonal entries of U (or R) above
      // threshold*max, the same criterion as Eigen's rank() and normalize_qb
      const RealType rankThreshold = std::numeric_limits<RealType>::epsilon() * std::max(m, r) * 10 * absDiag.maxCoeff();
      const int rank = (absDiag.array() > rankThreshold).count();
      if( rank < r )
      {
        // The caller must reduce the number of columns: with fewer selected rows A(I,:) is not square and
        // the interpolation coefficients A * A(I,:)^{-1} do not exist, so this cannot be fixed here.
        // Report what is needed to find the offending step (usually an orthogonal factor that lost its
        // orthogonality, see internal::tt_cross::dropNoiseDirections).
        throw std::invalid_argument("maxvol: input matrix (" + std::to_string(m) + " x " + std::to_string(r) +
                                    ") is (numerically) rank-deficient: estimated rank " + std::to_string(rank) +
                                    ", min/max of the absolute pivots " + std::to_string(absDiag.minCoeff()/absDiag.maxCoeff()) + "!");
      }

      // fast return: nothing to swap, all rows are selected. Return them in natural order so that
      // A * A(I,:)^{-1} is the identity with I[j] = j (with the pivot order it would be a permutation matrix)
      if( m == r )
      {
        for(int j = 0; j < r; j++)
          I[j] = j;
        if( AAinv != nullptr )
        {
          AAinv->resize(m, r);
          EigenMap(*AAinv) = mat::Identity(m, r);
        }
        return I;
      }

      // B = A * A(I,:)^{-1}, so B(I,:) is the identity
      Tensor2<T> B;
#ifdef PITTS_DIRECT_MKL_GEMM
      maxvol_lu_solve(LU, perm, B);
#else
      maxvol_solve(A, I, B);
#endif

      maxvol_swapRows(B, I, threshold, maxIter);

      // oversampling: add the rows with the largest scores |C(i,:)|^2 of C = A L with L L^H = (A(I,:)^H A(I,:))^{-1},
      // represented as C = B M with the small r x r matrix M (initially C = B), so B is not modified
      if( nRows > r )
      {
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> scores = ConstEigenMap(B).rowwise().squaredNorm();
        Tensor2<T> M(r, r);
        EigenMap(M) = mat::Identity(r, r);
        // workspace for maxvol_addRow
        Eigen::Matrix<T, 1, Eigen::Dynamic> c(r);
        Eigen::Matrix<T, Eigen::Dynamic, 1> w(r), Cc(m);
        for(int nSelected = r; nSelected < nRows; nSelected++)
        {
          for(const int i: I)
            scores(i) = -std::numeric_limits<RealType>::infinity();
          Eigen::Index i;
          scores.maxCoeff(&i);
          I.push_back(i);
          maxvol_addRow(i, B, M, scores, c, w, Cc);
        }
        if( AAinv != nullptr )
          maxvol_coefficients(B, M, I, *AAinv);
        return I;
      }

      // B is kept consistent with the final selection by the rank-1 updates, so it can be returned as is
      if( AAinv != nullptr )
        *AAinv = std::move(B);

      return I;
    }
  }
}


#endif // PITTS_TENSOR2_MAXVOL_IMPL_HPP
