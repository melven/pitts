/*! @file pitts_tensor3_split.hpp
* @brief split a simple rank-3 tensor into two (using a QR decomposition)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-22
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR3_SPLIT_HPP
#define PITTS_TENSOR3_SPLIT_HPP

// includes
#include <tuple>
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options
#include "pitts_tensor3.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! split a fixed-size rank-3 tensor into 2 smaller tensors
  //!
  //! Split t3c into t3a and t3b such that
  //!   t3c_(i,k,j) = sum_l t3a_(i,k1,l) * t3b_(l,k2,j)   with k=k2*na+k1
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param[in]  t3c       rank-3 tensor
  //! @param[in]  na        second dimension (n) of the first result tensor t3a
  //! @param[in]  nb        second dimension (n) of the second result tensor t3b
  //! @param[in]  leftOrtog make left part (t3a) orthogonal if true, otherwise t3b is made orthogonal
  //! @return     [t3a, t3b] the two parts of the splitted rank-3 tensor
  //!
  template<typename T>
  auto split(const Tensor3<T>& t3c, int na, int nb, bool leftOrthog = true)
  {
    const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

    using Matrix = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

    const auto r1 = t3c.r1();
    const auto r2 = t3c.r2();
    if( r1*r2 == 0 )
      throw std::invalid_argument("Unsupported dimension of zero!");

    if( na*nb != t3c.n() )
      throw std::invalid_argument("Invalid desired dimensions (na*na != t3c.n())!");

    Matrix t3cMat(r1*na, nb*r2);
    for(int i = 0; i < r1; i++)
      for(int j = 0; j < r2; j++)
        for(int k1 = 0; k1 < na; k1++)
          for(int k2 = 0; k2 < nb; k2++)
            t3cMat(i+k1*r1, k2+nb*j) = t3c(i, k1+na*k2, j);

    // use a faster QR algorithm (at the risk of a badly estimated truncation error)
    Matrix t3aMat, t3bMat;
    Eigen::Index r;
    if( leftOrthog )
    {
      Eigen::ColPivHouseholderQR<Matrix> qr(t3cMat);
      qr.setThreshold(1.e-10);
      r = std::max(Eigen::Index(1), qr.rank());

      // A P = Q R
      // => A = Q (R P^(-1))
      const Matrix Q = qr.matrixQ();
      const Matrix R = qr.matrixR().topRows(r).template triangularView<Eigen::Upper>();
      const auto P = qr.colsPermutation();
      t3aMat = Q.leftCols(r);
      t3bMat = R * P.inverse();
    }
    else // rightOrthog
    {
      Eigen::ColPivHouseholderQR<Matrix> qr(t3cMat.transpose());
      qr.setThreshold(1.e-10);
      r = std::max(Eigen::Index(1), qr.rank());

      // A^T P = Q R
      // => P^T A = R^T Q^T
      // => A = (R P^(-1))^T Q^T
      const Matrix Q = qr.matrixQ();
      const Matrix R = qr.matrixR().topRows(r).template triangularView<Eigen::Upper>();
      const auto P = qr.colsPermutation();
      t3aMat = (R * P.inverse()).transpose();
      t3bMat = Q.leftCols(r).transpose();
    }
    std::tuple<Tensor3<T>,Tensor3<T>> result;
    auto& [t3a, t3b] = result;
    t3a.resize(r1, na, r);
    t3b.resize(r, nb, r2);

    for(int i = 0; i < r1; i++)
      for(int j = 0; j < r; j++)
        for(int k = 0; k < na; k++)
          t3a(i,k,j) = t3aMat(i+r1*k,j);

    for(int i = 0; i < r; i++)
      for(int j = 0; j < r2; j++)
        for(int k = 0; k < nb; k++)
          t3b(i,k,j) = t3bMat(i,k+j*nb);

    return result;
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

}


#endif // PITTS_TENSOR3_SPLIT_HPP
