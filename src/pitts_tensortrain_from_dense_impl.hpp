/*! @file pitts_tensortrain_from_dense_impl.hpp
* @brief conversion of a dense tensor to the tensor-train format (based on a hopefully faster TSQR algorithm)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_FROM_DENSE_IMPL_HPP
#define PITTS_TENSORTRAIN_FROM_DENSE_IMPL_HPP

// includes
#include <numeric>
#include <iostream>
#include "pitts_tensortrain_from_dense.hpp"
#include "pitts_parallel.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_reshape.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_timer.hpp"
#include "pitts_eigen.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement TT fromDense
  template<typename T>
  TensorTrain<T> fromDense(MultiVector<T>& X, MultiVector<T>& work, const std::vector<int>& dimensions, T rankTolerance, int maxRank, bool mpiGlobal, int r0, int rd)
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

    const auto totalSize = r0 * rd * std::accumulate(begin(dimensions), end(dimensions), (std::ptrdiff_t)1, std::multiplies<std::ptrdiff_t>());
    const auto nDim = dimensions.size();
    if( X.rows()*X.cols() != totalSize )
      throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::fromDense");

    // for convenience, we also handle X.cols() == 1 != dims[-1]*rd with a warning
    if( X.cols() == 1 && dimensions[nDim-1]*rd != 1 )
    {
      // special case: nDim = 1
      if( nDim == 1 )
      {
        std::vector<Tensor3<T>> subT;
        // hopefully not strange for the caller that we steel its memory here
        subT.emplace_back(fold(std::move(X), r0, dimensions[0], rd));
        TensorTrain<T> result(std::move(subT));
        return result;
      }
      std::cout << "Warning: sub-optimal input dimension in fromDense, performing an additional copy...\n";
      std::swap(X, work);
      reshape(work, totalSize/(dimensions[nDim-1]*rd), dimensions[nDim-1]*rd, X);
    }

    if( X.rows() != totalSize/(dimensions[nDim-1]*rd) || X.cols() != dimensions[nDim-1]*rd )
      throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::fromDense");

    bool root = true;
    if( mpiGlobal )
    {
      const auto& [iProc,nProcs] = internal::parallel::mpiProcInfo();
      root = iProc == 0;
    }

    // actually convert to tensor train format
    Tensor2<T> tmpR;
    std::vector<Tensor3<T>> subTensors(nDim);
    using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
#if EIGEN_VERSION_AT_LEAST(3,4,90)
    Eigen::BDCSVD<EigenMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV> svd;
#else
    Eigen::BDCSVD<EigenMatrix> svd;
#endif
    using RealType = decltype(std::abs(T(1)));
    const RealType rankTol = std::abs(rankTolerance) / std::sqrt(RealType(nDim-1));
    RealType initialFrobeniusNorm = T(0);
    for(int iDim = nDim-1; iDim > 0; iDim--)
    {
      if( root )
        std::cout << "iDim: " << iDim << ", matrix dimensions: " << X.rows() << " x " << X.cols() << "\n";
      // calculate QR decomposition
      block_TSQR(X, tmpR, 0, mpiGlobal);
//std::cout << "tmpR:\n" << ConstEigenMap(tmpR) << "\n";

      // calculate SVD of R
#if EIGEN_VERSION_AT_LEAST(3,4,90)
      svd.compute(ConstEigenMap(tmpR));
#else
      svd.compute(ConstEigenMap(tmpR), Eigen::ComputeThinU | Eigen::ComputeThinV);
#endif
      if( iDim == nDim-1 )
        initialFrobeniusNorm = svd.singularValues().norm();
      int rank = internal::rankInFrobeniusNorm(svd, rankTol * initialFrobeniusNorm);
      if( maxRank >= 0 )
        rank = std::min(rank, maxRank);
      if( root )
        std::cout << "singular values: " << svd.singularValues().transpose() << "\n";

      // copy V to the TT sub-tensor
      fold_right(svd.matrixV().leftCols(rank).transpose(), dimensions[iDim], subTensors[iDim]);

      tmpR.resize(X.cols(), rank);
      EigenMap(tmpR) = svd.matrixV().leftCols(rank);

      const auto nextDim = dimensions[iDim-1];
      transform(X, tmpR, work, {X.rows()/nextDim, rank*nextDim});
      std::swap(X, work);
    }
    // last sub-tensor is now in X
    fold_right(X, dimensions[0], subTensors[0]);

    TensorTrain<T> result(dimensions);
    result.setSubTensors(0, std::move(subTensors));

    // make sure we swap X and work back: prevents problems where the reserved space in X is used again later AND the data does only fit into memory once ;)
    if( nDim % 2 == 0 )
      std::swap(X, work);

    return result;
  }

}

#endif // PITTS_TENSORTRAIN_FROM_DENSE_IMPL_HPP
