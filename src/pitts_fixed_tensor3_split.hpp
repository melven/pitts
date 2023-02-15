/*! @file pitts_fixed_tensor3_split.hpp
* @brief split a fixed-dimension rank-3 tensor into two (using a QR decomposition)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-29
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_tensor3_split.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_FIXED_TENSOR3_SPLIT)
import pitts_fixed_tensor3_split;
#define PITTS_FIXED_TENSOR3_SPLIT_HPP
#endif

// include guard
#ifndef PITTS_FIXED_TENSOR3_SPLIT_HPP
#define PITTS_FIXED_TENSOR3_SPLIT_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include <array>
#ifndef PITTS_USE_MODULES
#include "pitts_eigen.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#else
#include <complex>
#include <string>
#define EIGEN_CORE_MODULE_H
#include <Eigen/src/Core/util/Macros.h>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Core/util/ForwardDeclarations.h>
#endif
#include "pitts_fixed_tensor3.hpp"
#include "pitts_timer.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_fixed_tensor3_split;
# define PITTS_MODULE_EXPORT export
#endif

//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
{
  //! split a fixed-size rank-3 tensor into 2 smaller tensors
  //!
  //! Split t3c into t3a and t3b such that
  //!   t3c_(i,k,j) = sum_l t3a_(i,k1,l) * t3b_(l,k2,j)   with k=k2*N+k1
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  dimension
  //!
  //! @param[in]  t3c       rank-3 tensor
  //! @param[out] t3a       first part of splitted rank-3 tensor
  //! @param[out] t3b       second part of splitted rank-3 tensor
  //! @param[in]  leftOrtog make left part (t3a) orthogonal if true, otherwise t3b is made orthogonal
  //!
  template<typename T, int N>
  void split(const FixedTensor3<T,N*N>& t3c, FixedTensor3<T,N>& t3a, FixedTensor3<T,N>& t3b, bool leftOrthog = true)
  {
    const auto timer = PITTS::timing::createScopedTimer<FixedTensor3<T,N>>();

    using Matrix = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;
    using Stride = Eigen::OuterStride<Eigen::Dynamic>;
    using Map = Eigen::Map<Matrix, Eigen::Aligned128, Stride>;
    using ConstMap = Eigen::Map<const Matrix, Eigen::Aligned128, Stride>;

    const auto r1 = t3c.r1();
    const auto r2 = t3c.r2();
    if( r1*r2 == 0 )
      throw std::invalid_argument("Unsupported dimension of zero!");

    const auto t3cMap = ConstMap(&t3c(0,0,0), N*r1, N*r2, Stride(N*r1));

    // use a faster QR algorithm (at the risk of a badly estimated truncation error)
    if( leftOrthog )
    {
      Eigen::ColPivHouseholderQR<Matrix> qr(t3cMap);
      qr.setThreshold(1.e-10);
      using Index = decltype(qr.rank());
      const auto r = std::max(Index(1), qr.rank());

      t3a.resize(r1,r);
      t3b.resize(r,r2);

      // A P = Q R
      // => A = Q (R P^(-1))
      const Matrix Q = qr.matrixQ();
      const Matrix R = qr.matrixR().topRows(r).template triangularView<Eigen::Upper>();
      const auto P = qr.colsPermutation();
      Map(&t3a(0,0,0), r1*N, r, Stride(r1*N)) = Q.leftCols(r);
      Map(&t3b(0,0,0), r, r2*N, Stride(r)) = R * P.inverse();
    }
    else // rightOrthog
    {
      Eigen::ColPivHouseholderQR<Matrix> qr(t3cMap.transpose());
      qr.setThreshold(1.e-10);
      using Index = decltype(qr.rank());
      const auto r = std::max(Index(1), qr.rank());

      t3a.resize(r1,r);
      t3b.resize(r,r2);

      // A^T P = Q R
      // => P^T A = R^T Q^T
      // => A = (R P^(-1))^T Q^T
      const Matrix Q = qr.matrixQ();
      const Matrix R = qr.matrixR().topRows(r).template triangularView<Eigen::Upper>();
      const auto P = qr.colsPermutation();
      Map(&t3a(0,0,0), r1*N, r, Stride(r1*N)) = (R * P.inverse()).transpose();
      Map(&t3b(0,0,0), r, r2*N, Stride(r)) = Q.leftCols(r).transpose();
    }
    /*
    auto svd = Eigen::JacobiSVD<Matrix, Eigen::HouseholderQRPreconditioner>(t3cMap, Eigen::ComputeThinV | Eigen::ComputeThinU);
    svd.setThreshold(1.e-10);
    const auto r = svd.rank();

    t3a.resize(r1,r);
    t3b.resize(r,r2);
    if( leftOrthog )
    {
      Map(&t3a(0,0,0), r1*N,r) = svd.matrixU().leftCols(r);
      Map(&t3b(0,0,0), r,r2*N) = svd.singularValues().head(r).asDiagonal() * svd.matrixV().leftCols(r).adjoint();
    }
    else
    {
      Map(&t3a(0,0,0), r1*N,r) = svd.matrixU().leftCols(r) * svd.singularValues().head(r).asDiagonal();
      Map(&t3b(0,0,0), r,r2*N) = svd.matrixV().leftCols(r).adjoint();
    }
    //std::cout << "Singular value |.|: " << svd.singularValues().head(r).transpose().array().abs() << "\n";
    */
  }

  // explicit template instantiations
}


#endif // PITTS_FIXED_TENSOR3_SPLIT_HPP
