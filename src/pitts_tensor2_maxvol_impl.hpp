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
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include "pitts_tensor2_maxvol.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_eigen.hpp"
#include "pitts_timer.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    // implement maxvol
    template<typename T>
    std::vector<int> maxvol(const ConstTensor2View<T>& A, T tol, int maxIter)
    {
      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      using RealType = decltype(std::abs(T()));
      using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      using vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

      const int m = A.r1();
      const int r = A.r2();
      if( r < 1 || m < r )
        throw std::invalid_argument("maxvol: requires a tall matrix (m x r) with m >= r >= 1!");

      const RealType threshold = 1 + std::real(tol);
      const auto mapA = ConstEigenMap(A);

      // initial selection: column-pivoted QR of A^T (greedy choice of the "largest" rows)
      const Eigen::ColPivHouseholderQR<mat> qr(mapA.adjoint());
      const auto& pivots = qr.colsPermutation().indices();
      std::vector<int> I(r);
      for(int j = 0; j < r; j++)
        I[j] = pivots(j);

      // check for a rank-deficient input (A(I,:) would be singular)
      const RealType rankTolerance = std::numeric_limits<RealType>::epsilon() * std::max(m, r) * 10;
      const mat Rqr = qr.matrixR().template topLeftCorner(r, r).template triangularView<Eigen::Upper>();
      const RealType maxDiag = Rqr.diagonal().array().abs().maxCoeff();
      for(int j = 0; j < r; j++)
        if( std::abs(Rqr(j,j)) <= rankTolerance * maxDiag )
          throw std::invalid_argument("maxvol: input matrix is (numerically) rank-deficient!");

      // fast return: nothing to swap
      if( m == r )
        return I;

      // B = A * A(I,:)^{-1}, so B(I,:) is the identity
      mat Ahat(r, r);
      for(int j = 0; j < r; j++)
        Ahat.row(j) = mapA.row(I[j]);
      // (solve Ahat^H B^H = A^H for B^H)
      const Eigen::PartialPivLU<mat> luH(Ahat.adjoint());
      mat B = luH.solve(mapA.adjoint()).adjoint();

      for(int iter = 0; iter < maxIter; iter++)
      {
        // find the largest entry of B
        Eigen::Index i, j;
        const RealType maxB = B.array().abs().maxCoeff(&i, &j);
        if( maxB <= threshold )
          break;

        // swap row i into the selection (replacing the j-th selected row), rank-1 update of B:
        // B <- B - B(:,j) * (B(i,:) - e_j^T) / B(i,j)
        const T pivot = B(i,j);
        const vec Bj = B.col(j);
        Eigen::Matrix<T, 1, Eigen::Dynamic> Bi = B.row(i);
        Bi(j) -= T(1);
        B.noalias() -= Bj * (Bi / pivot);
        I[j] = i;
      }

      return I;
    }
  }
}


#endif // PITTS_TENSOR2_MAXVOL_IMPL_HPP
