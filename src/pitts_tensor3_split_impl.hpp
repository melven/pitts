/*! @file pitts_tensor3_split_impl.hpp
* @brief split a simple rank-3 tensor into two (using a QR decomposition)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-22
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR3_SPLIT_IMPL_HPP
#define PITTS_TENSOR3_SPLIT_IMPL_HPP

// includes
#include <tuple>
#include "pitts_tensor3_split.hpp"
#include "pitts_eigen.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_triangular_solve.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {

    //! small wrapper around QR only with data size
    template<typename T>
    auto normalize_qr_only(const Tensor2<T>& M, bool leftOrthog)
    {
      const auto n = M.r1();
      const auto m = M.r2();
      // 2n^2m-2/3n^3 for m>=n flops reported by LAPACKE
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"n", "m"},{n, m}}, // arguments
        {{(std::min(m,n)*std::min(n,m)*(3*std::max(m,n)-std::min(m,n))/3)*kernel_info::FMA<T>()}, // flops
         {(n*m)*kernel_info::Load<T>() + ((n+m+1)*std::min(n,m))*kernel_info::Store<T>()}} // data transfers
        );

      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      const auto mapM = ConstEigenMap(M);

      auto qr = leftOrthog ?
        Eigen::ColPivHouseholderQR<EigenMatrix>(mapM) :
        Eigen::ColPivHouseholderQR<EigenMatrix>(mapM.transpose());

      return qr;
    }

    //! small wrapper around SVD only with data size
    template<typename T>
    auto normalize_svd_only(const Tensor2<T>& M)
    {
      const auto n = M.r1();
      const auto m = M.r2();
      // 6.67 N^3 flops reported by LAPACK, round it to 7
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"n", "m"},{n, m}}, // arguments
        {{(7*n*m*std::min(n,m))/2*kernel_info::FMA<T>()}, // flops
         {(n*m)*kernel_info::Load<T>() + ((n+m+1)*std::min(n,m))*kernel_info::Store<T>()}} // data transfers
        );

      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

#if EIGEN_VERSION_AT_LEAST(3,4,90)
      auto svd = Eigen::BDCSVD<EigenMatrix, Eigen::ComputeThinV | Eigen::ComputeThinU>(ConstEigenMap(M));
#else
      auto svd = Eigen::BDCSVD<EigenMatrix>(ConstEigenMap(M), Eigen::ComputeThinV | Eigen::ComputeThinU);
#endif

      return svd;
    }

    // just for timings
    template<typename T>
    void normalize_qb_block_TSQR(const MultiVector<T>& X, Tensor2<T>& R)
    {
      const auto timer = PITTS::timing::createScopedTimer<MultiVector<T>>();

      block_TSQR(X, R, 0, false);
    }

    // implement normalize_qb
    template<typename T>
    std::pair<Tensor2<T>, Tensor2<T>> normalize_qb(const Tensor2<T>& M, bool leftOrthog, T rankTolerance, int maxRank, bool absoluteTolerance)
    {
      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      // get reasonable rank tolerance
      auto rankTol = std::abs(rankTolerance);
      rankTol = std::max(rankTol, std::numeric_limits<decltype(rankTol)>::epsilon() * std::min(M.r1(),M.r2()));

      //    M = Q R = Q U S V^T
      // => M = (QU) (SV^T)
      // => (QU) = M V S^(-1)
      MultiVector<T> mv;
      if( leftOrthog )
      {
        mv.resize(M.r1(), M.r2());
        EigenMap(mv) = ConstEigenMap(M);
      }
      else // rightOrthog
      {
        mv.resize(M.r2(), M.r1());
        EigenMap(mv) = ConstEigenMap(M).transpose();
      }
      Tensor2<T> R;
      normalize_qb_block_TSQR(mv, R);

      auto qr = normalize_qr_only(R, true);

      // we might want to consider an absolute tolerance, e.g. when orthogonalizing w.r.t. another set of orthogonal vectors...
      if( absoluteTolerance && qr.maxPivot() > 0 )
        rankTol /= qr.maxPivot();
      qr.setThreshold(rankTol);

      // with an absolute tolerance, we can get rank 0, otherwise it should be (numerically) at least 1
      // (Eigen / LAPACK / MKL don't like call with dimension zero, so avoid this when possible)
      const auto minRank = absoluteTolerance ? 0 : 1;
      using Index = decltype(qr.rank());
      const auto r = std::max(Index(minRank), std::min(qr.rank(), Index(maxRank)));

      // block_TSQR introduces an error of sqrt(numeric_limits<T>::min())
      using RealType = decltype(std::abs(T(0)));
      const auto tsqrError = std::sqrt(std::numeric_limits<RealType>::min());
      if( std::abs(qr.maxPivot()) < 1000*tsqrError )
      {
        // this is actually a zero input...
        std::pair<Tensor2<T>,Tensor2<T>> result;
        result.first.resize(M.r1(), minRank);
        result.second.resize(minRank, M.r2());
        if( minRank == 1 )
        {
          if( leftOrthog )
          {
            // return QR = e_1 * 0
            EigenMap(result.first) = EigenMatrix::Identity(M.r1(), 1);
            EigenMap(result.second) = EigenMatrix::Zero(1, M.r2());
          }
          else
          {
            // return LQ = 0 * e_1^T
            EigenMap(result.first) = EigenMatrix::Zero(M.r1(), 1);
            EigenMap(result.second) = EigenMatrix::Identity(1, M.r2());
          }
        }
        return result;
      }

      std::vector<int> colsPermutation(r);
      Eigen::Map<Eigen::VectorXi>(colsPermutation.data(), r) = qr.colsPermutation().indices().head(r);
      Tensor2<T> tmpR(r, r);
      EigenMap(tmpR) = qr.matrixR().topLeftCorner(r,r).template triangularView<Eigen::Upper>();
      triangularSolve(mv, tmpR, colsPermutation);
      /*
      auto mvMap = EigenMap(mv);
      mvMap.leftCols(r) = (mvMap * qr.colsPermutation()).leftCols(r);
      mvMap.leftCols(r) = qr.matrixR().topLeftCorner(r,r).template triangularView<Eigen::Upper>().template solve<Eigen::OnTheRight>(mvMap.leftCols(r));
      */

      std::pair<Tensor2<T>,Tensor2<T>> result;
      result.first.resize(M.r1(), r);
      result.second.resize(r, M.r2());
      if( r > 0 )
      {
        if( leftOrthog )
        {
          // return QR
          // X = QR,  RP = Q_2 R_2  =>  X P = Q Q_2 R_2  => X P R_2^(-1) = Q Q_2
          EigenMap(result.first) = ConstEigenMap(mv).leftCols(r);
          auto B = EigenMap(result.second);
          B = qr.matrixR().topRows(r).template triangularView<Eigen::Upper>();
          B = B * qr.colsPermutation().transpose();
        }
        else
        {
          // return LQ
          // (Q Q_2)^T = R_2^(-T) P^T X^T
          auto B = EigenMap(result.first);
          B = qr.matrixR().topRows(r).template triangularView<Eigen::Upper>().transpose();
          B = qr.colsPermutation() * B;
          EigenMap(result.second) = ConstEigenMap(mv).leftCols(r).transpose();
        }
      }

      return result;
    }

    // just for timings
    template<typename T>
    void normalize_svd_block_TSQR(const MultiVector<T>& X, Tensor2<T>& R)
    {
      const auto timer = PITTS::timing::createScopedTimer<MultiVector<T>>();

      block_TSQR(X, R, 0, false);
    }

    // implement normalize_svd
    template<typename T>
    std::pair<Tensor2<T>, Tensor2<T>> normalize_svd(const Tensor2<T>& M, bool leftOrthog, T rankTolerance, int maxRank)
    {

      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      // get reasonable rank tolerance
      auto rankTol = std::abs(rankTolerance);
      rankTol = std::max(rankTol, std::numeric_limits<decltype(rankTol)>::epsilon() * std::min(M.r1(),M.r2()));

      //    M = Q R = Q U S V^T
      // => M = (QU) (SV^T)
      // => (QU) = M V S^(-1)
      MultiVector<T> mv;
      if( leftOrthog )
      {
        mv.resize(M.r1(), M.r2());
        EigenMap(mv) = ConstEigenMap(M);
      }
      else // rightOrthog
      {
        mv.resize(M.r2(), M.r1());
        EigenMap(mv) = ConstEigenMap(M).transpose();
      }
      Tensor2<T> R;
      normalize_svd_block_TSQR(mv, R);

      auto svd = normalize_svd_only(R);

      svd.setThreshold(rankTol);
      using Index = decltype(svd.rank());
      const auto r = std::max(Index(1), std::min(svd.rank(), Index(maxRank)));

      // block_TSQR introduces an error of sqrt(numeric_limits<T>::min())
      using RealType = decltype(std::abs(T(0)));
      const auto tsqrError = std::sqrt(std::numeric_limits<RealType>::min());
      if( std::abs(svd.singularValues()(0)) < 1000*tsqrError )
      {
        // this is actually a zero input...
        std::pair<Tensor2<T>,Tensor2<T>> result;
        result.first.resize(M.r1(), 1);
        result.second.resize(1, M.r2());
        if( leftOrthog )
        {
          // return QB = e_1 * 0
          EigenMap(result.first) = EigenMatrix::Identity(M.r1(), 1);
          EigenMap(result.second) = EigenMatrix::Zero(1, M.r2());
        }
        else
        {
          // return BQ = 0 * e_1^T
          EigenMap(result.first) = EigenMatrix::Zero(M.r1(), 1);
          EigenMap(result.second) = EigenMatrix::Identity(1, M.r2());
        }
        return result;
      }

      return [&]()
      {
        const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"r1", "r", "r2", "leftOrthog"},{M.r1(), r, M.r2(), leftOrthog}}, // arguments
        {{M.r1()*M.r2()*r*kernel_info::FMA<T>()}, // flops
         {(M.r1()*M.r2()+r*M.r1()+r*M.r2())*kernel_info::Load<T>() + (M.r1()*r+M.r2()*r)*kernel_info::Store<T>()}} // data transfers
        );

        std::pair<Tensor2<T>,Tensor2<T>> result;
        result.first.resize(M.r1(), r);
        result.second.resize(r, M.r2());
        if( leftOrthog )
        {
          // return QB
          EigenMap(result.first) = ConstEigenMap(mv) * (svd.matrixV().leftCols(r) * svd.singularValues().head(r).array().inverse().matrix().asDiagonal());
          EigenMap(result.second) = svd.singularValues().head(r).asDiagonal() * svd.matrixV().leftCols(r).adjoint();
        }
        else
        {
          // return BQ
          EigenMap(result.first) = svd.matrixV().leftCols(r) * svd.singularValues().head(r).asDiagonal();
          EigenMap(result.second) = (svd.singularValues().head(r).array().inverse().matrix().asDiagonal() * svd.matrixV().leftCols(r).adjoint()) * ConstEigenMap(mv).transpose();
        }
        return result;
      }();
    }
  }


  // implement tensor3 split
  template<typename T>
  std::pair<Tensor3<T>, Tensor3<T>> split(const Tensor3<T>& t3c, int na, int nb, bool transpose, bool leftOrthog, T rankTolerance, int maxRank)
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
    t3cMat.resize(r1*na, nb*r2);
    for(int i = 0; i < r1; i++)
      for(int j = 0; j < r2; j++)
        for(int k1 = 0; k1 < na; k1++)
          for(int k2 = 0; k2 < nb; k2++)
            t3cMat(i+k1*r1, k2+nb*j) = transpose ? t3c(i, k2+nb*k1, j) : t3c(i, k1+na*k2, j);

    const auto [U,Vt] = rankTolerance != T(0) ?
      internal::normalize_svd(t3cMat, leftOrthog, rankTolerance, maxRank) :
      internal::normalize_qb(t3cMat, leftOrthog, rankTolerance, maxRank);

    std::pair<Tensor3<T>,Tensor3<T>> result;

    fold_left(U, na, result.first);
    fold_right(Vt, nb, result.second);

    return result;
  }

}


#endif // PITTS_TENSOR3_SPLIT_IMPL_HPP
