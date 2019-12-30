/*! @file pitts_fixed_tensor3_split.hpp
* @brief contract two simple fixed-dimension rank-3 tensors (along third and first dimension)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-29
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_FIXED_TENSOR3_SPLIT_HPP
#define PITTS_FIXED_TENSOR3_SPLIT_HPP

// includes
#include <array>
#include <Eigen/Dense>
#include "pitts_fixed_tensor3.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
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
  auto split(const FixedTensor3<T,N*N>& t3c, FixedTensor3<T,N>& t3a, FixedTensor3<T,N>& t3b, bool leftOrthog = true)
  {
    using Matrix = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;
    using Map = Eigen::Map<Matrix>;
    using ConstMap = Eigen::Map<const Matrix>;

    const auto r1 = t3c.r1();
    const auto r2 = t3c.r2();

    const auto t3cMap = ConstMap(&t3c(0,0,0), N*r1, N*r2);

    const auto svd = Eigen::JacobiSVD<Matrix>(t3cMap, Eigen::ComputeThinV | Eigen::ComputeThinU);
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
  }

}


#endif // PITTS_FIXED_TENSOR3_SPLIT_HPP
