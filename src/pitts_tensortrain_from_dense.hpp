/*! @file pitts_tensortrain_from_dense.hpp
* @brief conversion of a dense tensor to the tensor-train format (based on a hopefully faster TSQR algorithm)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_FROM_DENSE_HPP
#define PITTS_TENSORTRAIN_FROM_DENSE_HPP

// includes
#include "pitts_parallel.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_timer.hpp"
#include <limits>
#include <numeric>
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate tensor-train decomposition of a tensor stored in fully dense format
  //!
  //! Passing a large enough buffer in work helps to avoid costly reallocations + later page-faults for large data.
  //!
  //! @warning To reduce memory overhead, this function will overwrite the input arguments with temporary data.
  //!          Please pass a copy of the data if you still need it!
  //!
  //! @tparam T         underlying data type (double, complex, ...)
  //!
  //! @param X              input tensor, overwritten and modified output, dimension must be (size/lastDim, lastDim) where lastDim = dimensions.back()
  //! @param dimensions     tensor dimensions, input is interpreted in Fortran storage order (first index changes the fastest)
  //! @param work           buffer for temporary data, will be resized and modified
  //! @param rankTolerance  approximation accuracy, used to reduce the TTranks of the resulting tensor train
  //! @param maxRank        maximal TTrank (bond dimension), unbounded by default
  //! @param mpiGlobal      (experimental) perform a MPI parallel decomposition, this assumes that the data is distributed on the MPI processes and dimensions specify the local dimensions
  //! @param r0             first rank dimension of the first sub-tensor, handled like a zeroth dimension
  //! @param rd             last rank dimension of the last sub-tensors, handled like a (d+1)th dimension
  //! @return               resulting tensor train
  //!
  template<typename T>
  TensorTrain<T> fromDense(MultiVector<T>& X, MultiVector<T>& work, const std::vector<int>& dimensions, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = -1, bool mpiGlobal = false, int r0 = 1, int rd = 1)
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
    const auto nDims = dimensions.size();
    if( X.rows() != totalSize/(dimensions[nDims-1]*rd) || X.cols() != dimensions[nDims-1]*rd )
      throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::fromDense");

    bool root = true;
    if( mpiGlobal )
    {
      const auto& [iProc,nProcs] = internal::parallel::mpiProcInfo();
      root = iProc == 0;
    }

    TensorTrain<T> result(dimensions);

    // actually convert to tensor train format
    Tensor2<T> tmpR;
    using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::BDCSVD<EigenMatrix> svd;
    for(int iDim = nDims-1; iDim > 0; iDim--)
    {
      if( root )
        std::cout << "iDim: " << iDim << ", matrix dimensions: " << X.rows() << " x " << X.cols() << "\n";
      // calculate QR decomposition
      block_TSQR(X, tmpR, 0, mpiGlobal);
//std::cout << "tmpR:\n" << ConstEigenMap(tmpR) << "\n";

      // calculate SVD of R
      svd.compute(ConstEigenMap(tmpR), Eigen::ComputeThinU | Eigen::ComputeThinV);
      //Eigen::JacobiSVD<EigenMatrix> svd(ConstEigenMap(tmpR), Eigen::ComputeThinU | Eigen::ComputeThinV);
      svd.setThreshold(rankTolerance);
      if( root )
        std::cout << "singular values: " << svd.singularValues().transpose() << "\n";

      // copy V to the TT sub-tensor
      svd.setThreshold(rankTolerance);
      int rank = svd.rank();
      if( maxRank > 0 )
        rank = std::min(maxRank, rank);
      auto& subT = result.editableSubTensors()[iDim];
      subT.resize(rank, dimensions[iDim], X.cols()/dimensions[iDim]);
      for(int i = 0; i < subT.r1(); i++)
        for(int j = 0; j < subT.n(); j++)
          for(int k = 0; k < subT.r2(); k++)
            subT(i,j,k) = svd.matrixV()(j+subT.n()*k, i);

      tmpR.resize(X.cols(), rank);
      EigenMap(tmpR) = svd.matrixV().leftCols(rank);

      const auto nextDim = dimensions[iDim-1];
      transform(X, tmpR, work, {X.rows()/nextDim, rank*nextDim});
      std::swap(X, work);
    }
    // last sub-tensor is now in X
    auto& lastSubT = result.editableSubTensors()[0];
    lastSubT.resize(r0, dimensions[0], X.cols()/dimensions[0]);
    for(int i = 0; i < lastSubT.r1(); i++)
      for(int j = 0; j < lastSubT.n(); j++)
        for(int k = 0; k < lastSubT.r2(); k++)
        lastSubT(i, j, k) = X(i, j+k*dimensions[0]);

    // make sure we swap X and work back: prevents problems where the reserved space in X is used again later AND the data does only fit into memory once ;)
    if( nDims % 2 == 0 )
      std::swap(X, work);

    return result;
  }

}


#endif // PITTS_TENSORTRAIN_FROM_DENSE_HPP
