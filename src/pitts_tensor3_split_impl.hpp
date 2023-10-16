// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor3_split_impl.hpp
* @brief split a simple rank-3 tensor into two (using a QR decomposition)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-22
*
**/

// include guard
#ifndef PITTS_TENSOR3_SPLIT_IMPL_HPP
#define PITTS_TENSOR3_SPLIT_IMPL_HPP

// includes
#include <tuple>
#include "pitts_chunk_ops.hpp"
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
    auto normalize_qr_only(const ConstTensor2View<T>& M, bool leftOrthog)
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
    auto normalize_svd_only(const ConstTensor2View<T>& M, bool removeBottomZeroRows)
    {
      auto n = M.r1();
      const auto m = M.r2();

      // remove rows that are just (almost) zero (from TSQR)
      // circumvents Eigen bug: https://gitlab.com/libeigen/eigen/-/issues/2663
      using RealType = decltype(std::abs(T(0)));
      const auto tsqrError = std::sqrt(std::numeric_limits<RealType>::min());
      if( removeBottomZeroRows)
      {
        for(; n > 1; n--)
        {
          if( ConstEigenMap(M).row(n-1).norm() > 10*tsqrError )
            break;
        }
      }

      // 6.67 N^3 flops reported by LAPACK, round it to 7
      const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"n", "m"},{n, m}}, // arguments
        {{(7*n*m*std::min(n,m))/2*kernel_info::FMA<T>()}, // flops
         {(n*m)*kernel_info::Load<T>() + ((n+m+1)*std::min(n,m))*kernel_info::Store<T>()}} // data transfers
        );

      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;


      //std::cout << "SVD of:\n" << ConstEigenMap(M) << std::endl;
      const auto mapM = ConstEigenMap(M).topRows(n);

#if EIGEN_VERSION_AT_LEAST(3,4,90)
      auto svd = Eigen::BDCSVD<EigenMatrix, Eigen::ComputeThinV | Eigen::ComputeThinU>(mapM);
#else
      auto svd = Eigen::BDCSVD<EigenMatrix>(mapM, Eigen::ComputeThinV | Eigen::ComputeThinU);
#endif

      assert(!std::isnan(svd.singularValues()(0)));

      return svd;
    }

    // just for timings
    template<typename T>
    void normalize_qb_block_TSQR(const MultiVector<T>& X, Tensor2<T>& R)
    {
      const auto timer = PITTS::timing::createScopedTimer<MultiVector<T>>();

      block_TSQR(X, R, 0, false);
    }

    template<typename T>
    void normalize_qb_copy(const ConstTensor2View<T>& t2, MultiVector<T>& mv)
    {
      const auto n = t2.r1();
      const auto m = t2.r2();
      const auto timer = PITTS::performance::createScopedTimer<T>(
        {{"n", "m"},{n, m}}, // arguments
        {{n*m*kernel_info::NoOp<T>()}, // flops
         {(n*m)*kernel_info::Load<T>() + (n*m)*kernel_info::Store<T>()}} // data transfers
        );
      
      const long long cacheSize_L3 = 50000000;
      
      mv.resize(n, m);
      const int padding = 1 + (n-1) % Chunk<T>::size;
      const auto nChunks = mv.rowChunks();
#pragma omp parallel for collapse(2) schedule(static) if(m*nChunks > 50)
      for(long long j = 0; j < m; j++)
        for(long long iChunk = 0; iChunk < nChunks; iChunk++)
        {
          Chunk<T> tmp;
          if( iChunk+1 < nChunks )
            unaligned_load(&t2(iChunk*Chunk<T>::size,j), tmp);
          else // iChunk+1 == nChunks
          {
            for(int k = 0; k < padding; k++)
              tmp[k] = t2(iChunk*Chunk<T>::size+k,j);
            for(int k = padding; k < Chunk<T>::size; k++)
              tmp[k] = T(0);
          }
          if( n*m*sizeof(T) > cacheSize_L3 )
            streaming_store(tmp, mv.chunk(iChunk,j));
          else
            mv.chunk(iChunk,j) = tmp;
        }

      assert(ConstEigenMap(t2) == ConstEigenMap(mv));
    }

    template<typename T>
    void normalize_qb_copy(const MultiVector<T>& mv, Tensor2<T>& t2)
    {
      const auto n = mv.rows();
      const auto m = mv.cols();
      const auto timer = PITTS::performance::createScopedTimer<T>(
        {{"n", "m"},{n, m}}, // arguments
        {{n*m*kernel_info::NoOp<T>()}, // flops
         {(n*m)*kernel_info::Load<T>() + (n*m)*kernel_info::Store<T>()}} // data transfers
        );

      const long long cacheSize_L3 = 50000000;
      unsigned long long n_ = n, m_ = m;
      
      t2.resize(n, m);
      const unsigned long long nChunks = (n*m-1) / Chunk<T>::size + 1;
      Chunk<T>* t2data = (Chunk<T>*)&t2(0,0);
#pragma omp parallel for schedule(static) if(nChunks > 100)
      for(unsigned long long iChunk = 0; iChunk < nChunks; iChunk++)
      {
        Chunk<T> tmp;
        unsigned long long i = (iChunk*Chunk<T>::size) % n_;
        unsigned long long j = (iChunk*Chunk<T>::size) / n_;

        if( i+Chunk<T>::size <= n_)
          unaligned_load(&mv(i,j), tmp);
        else
        {
          for(int k = 0; k < Chunk<T>::size; k++)
          {
            unsigned long long i_ = (i + k) % n_;
            unsigned long long j_ = j + (i + k) / n_;
            tmp[k] = j_ < m_ ? mv(i_,j_) : T(0);
          }
        }

        if( n*m*sizeof(T) > cacheSize_L3 )
          streaming_store(tmp, t2data[iChunk]);
        else
          t2data[iChunk] = tmp;
      }
      
      assert(ConstEigenMap(t2) == ConstEigenMap(mv));
    }

    template<typename T>
    void normalize_qb_transpose(const ConstTensor2View<T>& t2, MultiVector<T>& mv)
    {
      const auto n = t2.r2();
      const auto m = t2.r1();
      const auto timer = PITTS::performance::createScopedTimer<T>(
        {{"n", "m"},{n, m}}, // arguments
        {{n*m*kernel_info::NoOp<T>()}, // flops
         {(n*m)*kernel_info::Load<T>() + (n*m)*kernel_info::Store<T>()}} // data transfers
        );
      
      const long long cacheSize_L3 = 50000000;
      
      mv.resize(n, m);
      const int padding = 1 + (n-1) % Chunk<T>::size;
      const auto nChunks = mv.rowChunks();
#pragma omp parallel for schedule(static) if(m*nChunks > 50)
      for(long long iChunk = 0; iChunk < nChunks; iChunk++)
        for(long long j = 0; j < m; j++)
        {
          Chunk<T> tmp;
          if( iChunk+1 < nChunks || padding == 0 )
          {
            for(int k = 0; k < Chunk<T>::size; k++)
              tmp[k] = t2(j,iChunk*Chunk<T>::size+k);
          }
          else // iChunk+1 == nChunks )
          {
            for(int k = 0; k < padding; k++)
              tmp[k] = t2(j,iChunk*Chunk<T>::size+k);
            for(int k = padding; k < Chunk<T>::size; k++)
              tmp[k] = T(0);
          }
          if( n*m*sizeof(T) > cacheSize_L3 )
            streaming_store(tmp, mv.chunk(iChunk,j));
          else
            mv.chunk(iChunk,j) = tmp;
        }

      assert(ConstEigenMap(t2) == ConstEigenMap(mv).transpose());
    }

    template<typename T>
    void normalize_qb_transpose(const MultiVector<T>& mv, Tensor2<T>& t2)
    {
      const auto n = mv.cols();
      const auto m = mv.rows();
      const auto timer = PITTS::performance::createScopedTimer<T>(
        {{"n", "m"},{n, m}}, // arguments
        {{n*m*kernel_info::NoOp<T>()}, // flops
         {(n*m)*kernel_info::Load<T>() + (n*m)*kernel_info::Store<T>()}} // data transfers
        );

      const long long cacheSize_L3 = 50000000;
      unsigned long long n_ = n, m_ = m;
      
      t2.resize(n, m);
#pragma omp parallel if(n*m > 100)
{
      for(long long i = 0; i < n; i+=Chunk<T>::size)
      {
#pragma omp for schedule(static) nowait
        for(long long j = 0; j < m; j++)
        {
          Chunk<T> tmp;
          if( i + Chunk<T>::size <= n )
          {
            for(int k = 0; k < Chunk<T>::size; k++)
              tmp[k] = mv(j,i+k);
            unaligned_store(tmp, &t2(i,j));
          }
          else
          {
            // last entries in each column
            for(int k = 0; i+k < n; k++)
              t2(i+k,j) = mv(j,i+k);
          }
        }
      }
}
      
      assert(ConstEigenMap(t2) == ConstEigenMap(mv).transpose());
    }

    // just for timings
    template<typename T>
    void normalize_qb_reorder(const auto& permutation, bool fromLeft, Tensor2<T>& B)
    {
      const auto n = B.r1();
      const auto m = B.r2();
      const auto timer = PITTS::performance::createScopedTimer<T>(
        {{"n", "m", "fromLeft"},{n, m, (int)fromLeft}}, // arguments
        {{n*m*kernel_info::NoOp<T>()}, // flops
         {(n*m)*kernel_info::Load<T>() + (n*m)*kernel_info::Store<T>()}} // data transfers
        );
      
      auto mapB = EigenMap(B);

      if( fromLeft )
        mapB = permutation * mapB;
      else
        mapB = mapB * permutation.transpose();
    }

    // implement normalize_qb
    template<typename T>
    std::pair<Tensor2<T>, Tensor2<T>> normalize_qb(const ConstTensor2View<T>& M, bool leftOrthog, T rankTolerance, int maxRank, bool absoluteTolerance)
    {
      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      // get reasonable rank tolerance
      auto rankTol = std::abs(rankTolerance);
      rankTol = std::max(rankTol, std::numeric_limits<decltype(rankTol)>::epsilon() * std::min(M.r1(),M.r2()));

#ifndef PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB
      //    M = Q R = Q U S V^T
      // => M = (QU) (SV^T)
      // => (QU) = M V S^(-1)
      MultiVector<T> mv;
      if( leftOrthog )
      {
        mv.resize(M.r1(), M.r2());
        normalize_qb_copy(M, mv);
      }
      else // rightOrthog
      {
        mv.resize(M.r2(), M.r1());
        normalize_qb_transpose(M, mv);
      }
      Tensor2<T> R;
      normalize_qb_block_TSQR(mv, R);

      auto qr = normalize_qr_only(R, true);
#else
      auto qr = normalize_qr_only(M, leftOrthog);
#endif

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
      if( std::abs(qr.maxPivot()) < 1000*tsqrError || qr.rank() == 0 || r == 0 )
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
#ifndef PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB
      triangularSolve(mv, tmpR, colsPermutation);
#endif

//std::cout << "QB rank: " << r << ", maxPivot / minPivot: " << std::abs(qr.matrixR()(0,0) / qr.matrixR()(r-1,r-1)) << ", diagonal entries of R: " << qr.matrixR().topLeftCorner(r,r).diagonal().transpose() << "\n";
//std::cout << "QB orthogonalityError: " << (ConstEigenMap(mv).transpose() * ConstEigenMap(mv) - EigenMatrix::Identity(r,r)).array().abs().maxCoeff() << "\n";
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
#ifndef PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB
          normalize_qb_copy(mv, result.first);
#else
          EigenMap(result.first) = qr.householderQ() * EigenMatrix::Identity(M.r1(), r);
#endif
          auto B = EigenMap(result.second);
          B = qr.matrixR().topRows(r).template triangularView<Eigen::Upper>();
          normalize_qb_reorder(qr.colsPermutation(), false, result.second);
        }
        else
        {
          // return LQ
          // (Q Q_2)^T = R_2^(-T) P^T X^T
          auto B = EigenMap(result.first);
          B = qr.matrixR().topRows(r).template triangularView<Eigen::Upper>().transpose();
          normalize_qb_reorder(qr.colsPermutation(), true, result.first);
#ifndef PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB
          normalize_qb_transpose(mv, result.second);
#else
          EigenMap(result.second) = (qr.householderQ() * EigenMatrix::Identity(M.r2(), r)).transpose();
#endif
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
    std::pair<Tensor2<T>, Tensor2<T>> normalize_svd(const ConstTensor2View<T>& M, bool leftOrthog, T rankTolerance, int maxRank, bool absoluteTolerance, bool useFrobeniusNorm, T* oldFrobeniusNorm)
    {
      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      using RealType = decltype(std::abs(T(1)));

      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

#ifndef PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB
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

      auto svd = normalize_svd_only(R, true);
#else
      auto svd = normalize_svd_only(M, false);
#endif

      if( oldFrobeniusNorm != nullptr )
        *oldFrobeniusNorm = svd.singularValues().norm();

      // block_TSQR introduces an error of sqrt(numeric_limits<T>::min())
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

      using Index = decltype(svd.rank());
      Index r = svd.rank();
      RealType rankTol = std::abs(rankTolerance);
      if( useFrobeniusNorm )
      {
        if( !absoluteTolerance )
          rankTol *= svd.singularValues().norm();
        r = rankInFrobeniusNorm(svd, rankTol);
      }
      else
      {
        if( absoluteTolerance )
          rankTol /= svd.singularValues()(0);
        rankTol = std::max(rankTol, std::numeric_limits<decltype(rankTol)>::epsilon() * std::min(M.r1(),M.r2()));

        svd.setThreshold(rankTol);
        r = svd.rank();
      }
      r = std::max(Index(1), std::min(r, Index(maxRank)));

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
#ifndef PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB
          EigenMap(result.first).noalias() = ConstEigenMap(mv) * (svd.matrixV().leftCols(r) * svd.singularValues().head(r).array().inverse().matrix().asDiagonal());
          EigenMap(result.second).noalias() = svd.singularValues().head(r).asDiagonal() * svd.matrixV().leftCols(r).adjoint();
#else
          EigenMap(result.first) = svd.matrixU().leftCols(r);
          EigenMap(result.second) = svd.singularValues().head(r).asDiagonal() * svd.matrixV().leftCols(r).adjoint();
#endif
        }
        else
        {
          // return BQ
#ifndef PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB
          EigenMap(result.first).noalias() = svd.matrixV().leftCols(r) * svd.singularValues().head(r).asDiagonal();
          EigenMap(result.second).noalias() = (svd.singularValues().head(r).array().inverse().matrix().asDiagonal() * svd.matrixV().leftCols(r).adjoint()) * ConstEigenMap(mv).transpose();
#else
          EigenMap(result.first) = svd.matrixU().leftCols(r) * svd.singularValues().head(r).asDiagonal();
          EigenMap(result.second) = svd.matrixV().leftCols(r).adjoint();
#endif
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

    const auto r1 = t3c.r1();
    const auto r2 = t3c.r2();
    if( r1*r2 == 0 )
      throw std::invalid_argument("Unsupported dimension of zero!");

    if( na*nb != t3c.n() )
      throw std::invalid_argument("Invalid desired dimensions (na*na != t3c.n())!");

    Tensor2<T> t3cMat;
    ConstTensor2View<T> t3cView;
    if( transpose )
    {
      t3cMat.resize(r1*na, nb*r2);
#pragma omp parallel for collapse(2) schedule(static) if(r1*na*r2*nb > 500)
      for(int j = 0; j < r2; j++)
        for(int k2 = 0; k2 < nb; k2++)
          for(int k1 = 0; k1 < na; k1++)
            for(int i = 0; i < r1; i++)
              t3cMat(i+k1*r1, k2+nb*j) = t3c(i, k2+nb*k1, j);
      t3cView = t3cMat;
    }
    else
    {
      t3cView = ConstTensor2View<T>(const_cast<Chunk<T>*>(t3c.data()), r1*na, nb*r2);
    }

    auto [U,Vt] = rankTolerance != T(0) ?
      internal::normalize_svd(t3cView, leftOrthog, rankTolerance, maxRank) :
      internal::normalize_qb(t3cView, leftOrthog, rankTolerance, maxRank);

    std::pair<Tensor3<T>,Tensor3<T>> result;

    result.first = fold_left(std::move(U), na);
    result.second = fold_right(std::move(Vt), nb);

    return result;
  }

}


#endif // PITTS_TENSOR3_SPLIT_IMPL_HPP
