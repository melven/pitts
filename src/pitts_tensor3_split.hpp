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
#include <complex>
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options
#include "pitts_tensor3.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! wrapper for qr, allows to show timings
    template<typename T>
    auto normalize_qb(const Tensor2<T>& M, T rankTolerance = 0, int maxRank = std::numeric_limits<int>::max())
    {
      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      // get reasonable rank tolerance
      auto rankTol = std::abs(rankTolerance);
      rankTol = std::max(rankTol, std::numeric_limits<decltype(rankTol)>::epsilon() * std::min(M.r1(),M.r2()));

      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      Eigen::ColPivHouseholderQR<EigenMatrix> qr(ConstEigenMap(M));
      qr.setThreshold(rankTol);
      const auto r = std::max(Eigen::Index(1), std::min(qr.rank(), Eigen::Index(maxRank)));
      qr.householderQ().setLength(r);

      const EigenMatrix Q = qr.householderQ() * EigenMatrix::Identity(M.r1(), r);
      const EigenMatrix R = qr.matrixR().topRows(r).template triangularView<Eigen::Upper>();
      const EigenMatrix B = R * qr.colsPermutation().inverse();

      return std::make_pair(Q, B);
    }

    //! wrapper for truncated SVD, allows to show timings, directly combines singular values with lefT/right singular vectors
    template<typename T>
    auto normalize_svd(const Tensor2<T>& M, bool leftOrthog, T rankTolerance = 0, int maxRank = std::numeric_limits<int>::max())
    {
      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      // get reasonable rank tolerance
      auto rankTol = std::abs(rankTolerance);
      rankTol = std::max(rankTol, std::numeric_limits<decltype(rankTol)>::epsilon() * std::min(M.r1(),M.r2()));

      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      auto svd = Eigen::JacobiSVD<EigenMatrix, Eigen::HouseholderQRPreconditioner>(ConstEigenMap(M), Eigen::ComputeThinV | Eigen::ComputeThinU);
      svd.setThreshold(rankTol);
      const auto r = std::max(Eigen::Index(1), std::min(svd.rank(), Eigen::Index(maxRank)));
      EigenMatrix U, Vt;
      if( leftOrthog )
      {
        U = svd.matrixU().leftCols(r);
        Vt = svd.singularValues().head(r).asDiagonal() * svd.matrixV().leftCols(r).adjoint();
      }
      else
      {
        U = svd.matrixU().leftCols(r) * svd.singularValues().head(r).asDiagonal();
        Vt = svd.matrixV().leftCols(r).adjoint();
      }

      return std::make_pair(U, Vt);
    }
  }


  //! split a fixed-size rank-3 tensor into 2 smaller tensors
  //!
  //! Split t3c into t3a and t3b such that
  //!   t3c_(i,k,j) = sum_l t3a_(i,k1,l) * t3b_(l,k2,j)   with k=k2*na+k1
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param[in]  t3c           rank-3 tensor
  //! @param[in]  na            second dimension (n) of the first result tensor t3a
  //! @param[in]  nb            second dimension (n) of the second result tensor t3b
  //! @param[in]  leftOrtog     make left part (t3a) orthogonal if true, otherwise t3b is made orthogonal
  //! @param[in]  rankTolerance truncation tolerance for determining the rank between t3a and t3b
  //! @param[in]  maxRank       maximal rank of the decomposition of t3c into t3a and t3b
  //! @return     [t3a, t3b]    the two parts of the splitted rank-3 tensor
  //!
  template<typename T>
  auto split(const Tensor3<T>& t3c, int na, int nb, bool leftOrthog = true, T rankTolerance = 0, int maxRank = std::numeric_limits<int>::max())
  {
    const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

    using Matrix = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

    const auto r1 = t3c.r1();
    const auto r2 = t3c.r2();
    if( r1*r2 == 0 )
      throw std::invalid_argument("Unsupported dimension of zero!");

    if( na*nb != t3c.n() )
      throw std::invalid_argument("Invalid desired dimensions (na*na != t3c.n())!");

    Tensor2<T> t3cMat;
    if( leftOrthog )
    {
      t3cMat.resize(r1*na, nb*r2);
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < r2; j++)
          for(int k1 = 0; k1 < na; k1++)
            for(int k2 = 0; k2 < nb; k2++)
              t3cMat(i+k1*r1, k2+nb*j) = t3c(i, k1+na*k2, j);
    }
    else
    {
      t3cMat.resize(r2*nb, na*r1);
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < r2; j++)
          for(int k1 = 0; k1 < na; k1++)
            for(int k2 = 0; k2 < nb; k2++)
              t3cMat(j+k2*r2, k1+na*i) = t3c(i, k1+na*k2, j);
    }

    const auto [Q,B] = rankTolerance != T(0) ?
      internal::normalize_svd(t3cMat, true, rankTolerance, maxRank) :
      internal::normalize_qb(t3cMat, rankTolerance, maxRank);
    const int r = Q.cols();

    std::tuple<Tensor3<T>,Tensor3<T>> result;
    auto& [t3a, t3b] = result;
    t3a.resize(r1, na, r);
    t3b.resize(r, nb, r2);

    if( leftOrthog )
    {
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < r; j++)
          for(int k = 0; k < na; k++)
            t3a(i,k,j) = Q(i+r1*k,j);

      for(int i = 0; i < r; i++)
        for(int j = 0; j < r2; j++)
          for(int k = 0; k < nb; k++)
            t3b(i,k,j) = B(i,k+j*nb);
    }
    else
    {
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < r; j++)
          for(int k = 0; k < na; k++)
            t3a(i,k,j) = B(j,k+na*i);

      for(int i = 0; i < r; i++)
        for(int j = 0; j < r2; j++)
          for(int k = 0; k < nb; k++)
            t3b(i,k,j) = Q(j+k*r2,i);
    }

    return result;
  }

}


#endif // PITTS_TENSOR3_SPLIT_HPP