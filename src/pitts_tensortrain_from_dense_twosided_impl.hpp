/*! @file pitts_tensortrain_from_dense_twosided_impl.hpp
* @brief conversion of a dense tensor to the tensor-train format (based on a hopefully faster TSQR algorithm, two-sided variant)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-08-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_IMPL_HPP
#define PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_IMPL_HPP

// includes
#include <numeric>
#include <iostream>
#include "pitts_tensortrain_from_dense_twosided.hpp"
#include "pitts_eigen.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_transpose.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! helper namespace for high-order SVD functionality (e.g. TensorTrain_fromDense)
    namespace HOSVD
    {
      template<typename T>
      void split(const MultiVector<T>& X, MultiVector<T>& Y, Tensor2<T>& M, int nextDim, T rankTolerance, int maxRank, T& initialFrobeniusNorm)
      {
        using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
#if EIGEN_VERSION_AT_LEAST(3,4,90)
        Eigen::BDCSVD<EigenMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV> svd;
#else
        Eigen::BDCSVD<EigenMatrix> svd;
#endif
std::cout << "HOSVD::split  matrix dimensions: " << X.rows() << " x " << X.cols() << "\n";
        if( X.rows() > 10*X.cols() )
        {
          // calculate QR decomposition (QR-trick: X=QR,SVD(R))
          block_TSQR(X, M, 0, false);
#if EIGEN_VERSION_AT_LEAST(3,4,90)
          svd.compute(ConstEigenMap(M));
#else
          svd.compute(ConstEigenMap(M), Eigen::ComputeThinU | Eigen::ComputeThinV);
#endif
        }
        else
        {
#if EIGEN_VERSION_AT_LEAST(3,4,90)
          svd.compute(ConstEigenMap(X));
#else
          svd.compute(ConstEigenMap(X), Eigen::ComputeThinU | Eigen::ComputeThinV);
#endif
        }

        if( initialFrobeniusNorm == 0 )
          initialFrobeniusNorm = svd.singularValues().norm();

        int rank = internal::rankInFrobeniusNorm(svd, rankTolerance);
        if( maxRank >= 0 )
          rank = std::min(rank, maxRank);

        std::cout << "singular values: " << svd.singularValues().transpose() << "\n";

        // copy right singular vectors
        M.resize(X.cols(), rank);
        EigenMap(M) = svd.matrixV().leftCols(rank);

        // transform input, s.t. Y \approx X M
        transform(X, M, Y, {X.rows()/nextDim, rank*nextDim});
      }
    }
  }


  // implement TT fromDense twosided
  template<typename T>
  TensorTrain<T> fromDense_twoSided(MultiVector<T>& X, MultiVector<T>& work, const std::vector<int>& dimensions, T rankTolerance, int maxRank)
  {
    // timer
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // abort early for zero dimensions
    if( dimensions.size() == 0 )
    {
      if( X.rows()*X.cols() !=  0 )
        throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::fromDense");
      return TensorTrain<T>{dimensions};
    }

    const auto totalSize = std::accumulate(begin(dimensions), end(dimensions), (std::ptrdiff_t)1, std::multiplies<std::ptrdiff_t>());
    const auto nDim = dimensions.size();
    if( X.rows() != totalSize/dimensions[nDim-1] || X.cols() != dimensions[nDim-1] )
      throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::fromDense");

    // actually convert to tensor train format
    std::vector<Tensor3<T>> subTensors(nDim);
    Tensor2<T> M;
    using RealType = decltype(std::abs(T(1)));
    const RealType rankTol = std::abs(rankTolerance) / std::sqrt(RealType(nDim-1));
    RealType initialFrobeniusNorm = 0;
    for(int ii = 0; ii < nDim; ii++)
    {
      if( ii % 2 == 0 )
      {
        // right part
        const auto iDim = nDim - 1 - ii/2;
        if( ii > 0 )
        {
          const auto r1 = subTensors[iDim+1].r1();
          const auto n = dimensions[iDim];
          transpose(work, X, {(work.rows()*work.cols())/(n*r1), n*r1}, true);
        }
        if( ii != nDim-1 )
        {
          internal::HOSVD::split(X, work, M, 1, rankTolerance, maxRank, initialFrobeniusNorm);
        }
        else
        {
          M.resize(X.cols(), X.rows());
          EigenMap(M) = ConstEigenMap(X).transpose();
        }

        auto& subT = subTensors[iDim];
        const int rank = M.r2();
        subT.resize(rank, dimensions[iDim], X.cols()/dimensions[iDim]);
        for(int i = 0; i < rank; i++)
          for(int j = 0; j < subT.n(); j++)
            for(int k = 0; k < subT.r2(); k++)
              subT(i,j,k) = M(j+subT.n()*k, i);
      }
      else
      {
        // left part
        const auto iDim = ii / 2;
        {
          const auto r2 = (iDim == 0 ) ? 1 : subTensors[iDim-1].r2();
          const auto n = dimensions[iDim];
          transpose(work, X, {(work.cols()*work.rows())/(n*r2), n*r2}, false);
        }
        if( ii != nDim - 1 )
        {
          internal::HOSVD::split(X, work, M, 1, rankTolerance, maxRank, initialFrobeniusNorm);
        }
        else
        {
          M.resize(X.cols(), X.rows());
          EigenMap(M) = ConstEigenMap(X).transpose();
        }

        auto& subT = subTensors[iDim];
        const int rank = M.r2();
        subT.resize(X.cols()/dimensions[iDim], dimensions[iDim], rank);
        for(int i = 0; i < rank; i++)
          for(int j = 0; j < subT.n(); j++)
            for(int k = 0; k < subT.r1(); k++)
              subT(k,j,i) = M(k+j*subT.r1(), i);
      }
    }

    TensorTrain<T> result(dimensions);
    result.setSubTensors(0, std::move(subTensors));

    return result;
  }

}


#endif // PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_IMPL_HPP
