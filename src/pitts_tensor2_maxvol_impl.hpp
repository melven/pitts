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
    void maxvol_lu_solve(Tensor2<T>& LU, const std::vector<int>& perm, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& B)
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
      for(int k = 0; k < r; k++)
      {
        B.row(perm[k]).setZero();
        B(perm[k], k) = T(1);
      }
      for(int k = r; k < m; k++)
        B.row(perm[k]) = mapLU.row(k);
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
    auto maxvol_solve(const ConstTensor2View<T>& A, const std::vector<int>& I)
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
      return mat(luH.solve(mapA.adjoint()).adjoint());
    }
#endif

    //! find the entry of B with the largest absolute value (just for timings)
    template<typename T>
    auto maxvol_maxAbs(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& B, Eigen::Index& i, Eigen::Index& j)
    {
      const auto m = B.rows();
      const auto r = B.cols();
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r"},{m, r}}, // arguments
        {{(1.*m*r)*kernel_info::Add<T>()}, // flops (one comparison per entry)
         {(m*r)*kernel_info::Load<T>()}} // data transfers
        );

      return B.array().abs().maxCoeff(&i, &j);
    }

    //! rank-1 update of B = A * A(I,:)^{-1} when row i replaces the j-th selected row (just for timings)
    //!
    //! B <- B - B(:,j) * (B(i,:) - e_j^T) / B(i,j), so B(I,:) stays the identity for the new selection.
    //!
    template<typename T>
    void maxvol_swapRow(Eigen::Index i, Eigen::Index j, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& B)
    {
      const auto m = B.rows();
      const auto r = B.cols();
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r"},{m, r}}, // arguments
        {{(1.*m*r)*kernel_info::FMA<T>()}, // flops
         {(m*r)*kernel_info::Update<T>()}} // data transfers
        );

      using vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

      const T pivot = B(i,j);
      const vec Bj = B.col(j);
      Eigen::Matrix<T, 1, Eigen::Dynamic> Bi = B.row(i);
      Bi(j) -= T(1);
      B.noalias() -= Bj * (Bi / pivot);
    }

    //! copy the resulting B = A * A(I,:)^{-1} to the output argument (just for timings)
    template<typename T>
    void maxvol_copy(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& B, Tensor2<T>& AAinv)
    {
      const auto m = B.rows();
      const auto r = B.cols();
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r"},{m, r}}, // arguments
        {{0*kernel_info::Add<T>()}, // flops
         {(m*r)*kernel_info::Load<T>() + (m*r)*kernel_info::Store<T>()}} // data transfers
        );

      AAinv.resize(m, r);
      EigenMap(AAinv) = B;
    }

    //! add row i to an oversampled selection, with C = B M only the small matrix M is updated (just for timings)
    //!
    //! M <- M - beta (M c^H) c with c = C(i,:) = B(i,:) M, and the scores |C(j,:)|^2 are updated directly; B is only read.
    //!
    template<typename T, typename RealType>
    void maxvol_addRow(Eigen::Index i, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& B, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& M, Eigen::Matrix<RealType, Eigen::Dynamic, 1>& scores)
    {
      const auto m = B.rows();
      const auto r = B.cols();
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r"},{m, r}}, // arguments
        {{(1.*m*r + 3.*r*r)*kernel_info::FMA<T>()}, // flops
         {(m*r)*kernel_info::Load<T>() + m*kernel_info::Update<T>()}} // data transfers
        );

      using vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

      // C = A L is replaced by A L' with L' = L (I - beta c^H c), where (I - beta c^H c)^2 = I - c^H c / (1 + |c|^2), so that
      // L' L'^H = (G + u^H u)^{-1} for the Gram matrix G of the selection and the new row u (Sherman-Morrison);
      // beta = (1 - 1/sqrt(1+a))/a = 1/(sqrt(1+a) (1+sqrt(1+a))) with a = |c|^2, the second form is stable for small a.
      // With C = B M: C' = B M (I - beta c^H c) = B (M - beta w c) with w = M c^H, and
      // |C'(j,:)|^2 = C(j,:) (I - c^H c / (1+a)) C(j,:)^H = |C(j,:)|^2 - |(C c^H)_j|^2 / (1+a) with C c^H = B w.
      const Eigen::Matrix<T, 1, Eigen::Dynamic> c = B.row(i) * M;
      const RealType a = c.squaredNorm();
      const RealType sqrt1a = std::sqrt(1 + a);
      const RealType beta = 1 / (sqrt1a * (1 + sqrt1a));
      const vec w = M * c.adjoint();
      const vec Cc = B * w;
      scores -= Cc.cwiseAbs2() / (1 + a);
      M.noalias() -= (beta * w) * c;
    }

    //! interpolation coefficients C C(I,:)^H = A A(I,:)^+ of an oversampled selection with C = B M (just for timings)
    template<typename T>
    void maxvol_coefficients(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& B, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& M, const std::vector<int>& I, Tensor2<T>& AAinv)
    {
      const auto m = B.rows();
      const auto r = B.cols();
      const long long q = I.size();
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r", "q"},{m, r, q}}, // arguments
        {{(1.*m*r*q + 2.*r*r*q)*kernel_info::FMA<T>()}, // flops
         {(m*r)*kernel_info::Load<T>() + (m*q)*kernel_info::Store<T>()}} // data transfers
        );

      using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      // the selected rows as an explicit loop (an Eigen indexed view triggers -Warray-bounds false positives with GCC 13)
      mat BI(q, r);
      for(int k = 0; k < q; k++)
        BI.row(k) = B.row(I[k]);
      // C C(I,:)^H = B M (B(I,:) M)^H: first the small r x q factor, then one product with B
      const mat CI = BI * M;
      const mat K = M * CI.adjoint();
      AAinv.resize(m, q);
      EigenMap(AAinv).noalias() = B * K;
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
#ifdef PITTS_DIRECT_MKL_GEMM
      mat B;
      maxvol_lu_solve(LU, perm, B);
#else
      mat B = maxvol_solve(A, I);
#endif

      for(int iter = 0; iter < maxIter; iter++)
      {
        Eigen::Index i, j;
        const RealType maxB = maxvol_maxAbs(B, i, j);
        if( maxB <= threshold )
          break;

        // swap row i into the selection, replacing the j-th selected row
        maxvol_swapRow(i, j, B);
        I[j] = i;
      }

      // oversampling: add the rows with the largest scores |C(i,:)|^2 of C = A L with L L^H = (A(I,:)^H A(I,:))^{-1},
      // represented as C = B M with the small r x r matrix M (initially C = B), so B is not modified
      if( nRows > r )
      {
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> scores = B.rowwise().squaredNorm();
        mat M = mat::Identity(r, r);
        for(int nSelected = r; nSelected < nRows; nSelected++)
        {
          for(const int i: I)
            scores(i) = -std::numeric_limits<RealType>::infinity();
          Eigen::Index i;
          scores.maxCoeff(&i);
          I.push_back(i);
          maxvol_addRow(i, B, M, scores);
        }
        if( AAinv != nullptr )
          maxvol_coefficients(B, M, I, *AAinv);
        return I;
      }

      // B is kept consistent with the final selection by the rank-1 updates, so it can be returned as is
      if( AAinv != nullptr )
        maxvol_copy(B, *AAinv);

      return I;
    }
  }
}


#endif // PITTS_TENSOR2_MAXVOL_IMPL_HPP
