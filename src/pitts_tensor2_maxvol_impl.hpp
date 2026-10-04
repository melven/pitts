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


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
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

    //! add row i to an oversampled selection: C <- C - beta (C c^H) c with c = C(i,:), and new scores |C(j,:)|^2 (just for timings)
    template<typename T, typename RealType>
    void maxvol_addRow(Eigen::Index i, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& C, Eigen::Matrix<RealType, Eigen::Dynamic, 1>& scores)
    {
      const auto m = C.rows();
      const auto r = C.cols();
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r"},{m, r}}, // arguments
        {{(3.*m*r)*kernel_info::FMA<T>()}, // flops
         {(m*r)*kernel_info::Update<T>() + (m*r)*kernel_info::Load<T>()}} // data transfers
        );

      using vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

      // C = A L is replaced by A L' with L' = L (I - beta c^H c), where (I - beta c^H c)^2 = I - c^H c / (1 + |c|^2), so that
      // L' L'^H = (G + u^H u)^{-1} for the Gram matrix G of the selection and the new row u (Sherman-Morrison);
      // beta = (1 - 1/sqrt(1+a))/a = 1/(sqrt(1+a) (1+sqrt(1+a))) with a = |c|^2, the second form is stable for small a
      const Eigen::Matrix<T, 1, Eigen::Dynamic> c = C.row(i);
      const RealType sqrt1a = std::sqrt(1 + c.squaredNorm());
      const RealType beta = 1 / (sqrt1a * (1 + sqrt1a));
      const vec Cc = C * c.adjoint();
      C.noalias() -= (beta * Cc) * c;
      scores = C.rowwise().squaredNorm();
    }

    //! interpolation coefficients C C(I,:)^H = A A(I,:)^+ of an oversampled selection (just for timings)
    template<typename T>
    void maxvol_coefficients(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& C, const std::vector<int>& I, Tensor2<T>& AAinv)
    {
      const auto m = C.rows();
      const auto r = C.cols();
      const long long q = I.size();
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"m", "r", "q"},{m, r, q}}, // arguments
        {{(1.*m*r*q)*kernel_info::FMA<T>()}, // flops
         {(m*r)*kernel_info::Load<T>() + (m*q)*kernel_info::Store<T>()}} // data transfers
        );

      using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      // the selected rows as an explicit loop (an Eigen indexed view triggers -Warray-bounds false positives with GCC 13)
      mat CI(q, r);
      for(int k = 0; k < q; k++)
        CI.row(k) = C.row(I[k]);
      AAinv.resize(m, q);
      EigenMap(AAinv).noalias() = C * CI.adjoint();
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

      // initial selection: column-pivoted QR of A^T (greedy choice of the "largest" rows)
      auto qr = maxvol_qr(A);
      const auto& pivots = qr.colsPermutation().indices();
      std::vector<int> I(r);
      for(int j = 0; j < r; j++)
        I[j] = pivots(j);

      // check for a rank-deficient input (A(I,:) would be singular); Eigen's rank() counts the diagonal
      // entries of R above threshold*maxPivot, the same criterion as in normalize_qb
      qr.setThreshold(std::numeric_limits<RealType>::epsilon() * std::max(m, r) * 10);
      if( qr.rank() < r )
      {
        // The caller must reduce the number of columns: with fewer selected rows A(I,:) is not square and
        // the interpolation coefficients A * A(I,:)^{-1} do not exist, so this cannot be fixed here.
        // Report what is needed to find the offending step (usually an orthogonal factor that lost its
        // orthogonality, see internal::tt_cross::dropNoiseDirections).
        const auto absDiagR = qr.matrixR().diagonal().array().abs().eval();
        throw std::invalid_argument("maxvol: input matrix (" + std::to_string(m) + " x " + std::to_string(r) +
                                    ") is (numerically) rank-deficient: estimated rank " + std::to_string(qr.rank()) +
                                    ", |R_rr/R_11| " + std::to_string(absDiagR(r-1)/absDiagR(0)) + "!");
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
      mat B = maxvol_solve(A, I);

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

      // oversampling: add the rows with the largest scores |B(i,:)|^2, B is updated to C = A L with L L^H = (A(I,:)^H A(I,:))^{-1}
      if( nRows > r )
      {
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> scores = B.rowwise().squaredNorm();
        for(int nSelected = r; nSelected < nRows; nSelected++)
        {
          for(const int i: I)
            scores(i) = -std::numeric_limits<RealType>::infinity();
          Eigen::Index i;
          scores.maxCoeff(&i);
          I.push_back(i);
          maxvol_addRow(i, B, scores);
        }
        if( AAinv != nullptr )
          maxvol_coefficients(B, I, *AAinv);
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
