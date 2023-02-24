/*! @file pitts_tensor2_qb_decomposition.hpp
* @brief QB-part of the SVQB orthogonalization algorithm (Stathopoulos and Wu, SISC 23 (6), pp. 2165-2182)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-31
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR2_QB_DECOMPOSITION_HPP
#define PITTS_TENSOR2_QB_DECOMPOSITION_HPP

// includes
#include <exception>
#include <cmath>
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_timer.hpp"
#include "pitts_eigen.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! QB-part in the SVQB orthogonalization algorithm from Stathopoulos and Wu, SISC 23 (6), pp. 2165-2182
  //!
  //! Computes the decomposition B^TB = M using a SVD of M for a symmetric positive semi-definite matrix M
  //!
  //! @tparam T                 underlying data type (double, complex, ...)
  //!
  //! @param  M                 Symmetric positive semi-definite input matrix M
  //! @param  B                 Resulting matrix B with B^TB = M
  //! @param  Binv              Pseudo-Inverse of the output matrix B
  //! @param  rankTolerance     Desired accuracy / truncation tolerance
  //! @param  absoluteTolerance Use rankTolerance as absolute truncation tolerance instead of relative to the biggest singular value.
  //! @return                   detected rank of the matrix
  //!
  template<typename T>
  auto qb_decomposition(const Tensor2<T>& M, Tensor2<T>& B, Tensor2<T>& Binv, T rankTolerance, bool absoluteTolerance = false)
  {
    const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

    // get dimension
    if( M.r1() != M.r2() )
      throw std::invalid_argument("qb_decomposition requires a quadratic matrix!");
    const auto n = M.r1();

    // Helpful types
    using vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using EigenSolver = Eigen::SelfAdjointEigenSolver<mat>;

    // use Eigen for matrix operations
    const auto mapM = ConstEigenMap(M);

    // compute sqrt(diag(M)) and its inverse
    const auto maxDiag = ConstEigenMap(M).diagonal().array().abs().maxCoeff();
    const auto diagTol2 = rankTolerance*rankTolerance * (absoluteTolerance ? 1 : maxDiag);

    vec d(n), dinv(n);
    for(int i = 0; i < n; i++)
    {
      if( std::abs(M(i,i)) <= diagTol2 )
      {
        d(i) = T(0);
        dinv(i) = T(0);
      }
      else
      {
        d(i) = std::sqrt(M(i,i));
        dinv(i) = T(1)/d(i);
      }
    }

    // scale input matrix from left and right by diag(dinv), so it's diagonal entries become 1
    const mat scaledM = dinv.asDiagonal() * mapM * dinv.asDiagonal();

    // compute eigenvalue decomposition
    EigenSolver eigSolv(scaledM);

    // determine rank of the input matrix and set w = sqrt(lambda), winv = 1/w
    // Eigen orders eigenvalues with increasing value (smallest first)
    const auto evMax = std::abs(eigSolv.eigenvalues()(n-1));
    if( evMax <= T(0) )
    {
      B.resize(0,n);
      Binv.resize(n,0);
      return 0;
    }

    vec w(n), winv(n);
    int rank = n;
    const auto eigTol2 = diagTol2 / maxDiag;
    for(int i = 0; i < n; i++)
    {
      if( eigSolv.eigenvalues()(i) <= eigTol2 )
      {
        rank--;
        w(i) = T(0);
        winv(i) = T(0);
      }
      else
      {
        w(i) = std::sqrt(eigSolv.eigenvalues()(i));
        winv(i) = T(1)/w(i);
      }
    }

    // calculate B and Binv
    B.resize(rank,n);
    Binv.resize(n,rank);
    EigenMap(Binv) = dinv.asDiagonal() * eigSolv.eigenvectors().rightCols(rank) * winv.bottomRows(rank).asDiagonal();
    EigenMap(B) = w.bottomRows(rank).asDiagonal() * eigSolv.eigenvectors().rightCols(rank).transpose() * d.asDiagonal();

    return rank;
  }

}


#endif // PITTS_TENSOR2_QB_DECOMPOSITION_HPP
