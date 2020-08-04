/*! @file pitts_tensortrain_from_dense_tsqr.hpp * @brief conversion of a dense tensor to the tensor-train format (based on a hopefully faster TSQR algorithm)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_FROM_DENSE_TSQR_HPP
#define PITTS_TENSORTRAIN_FROM_DENSE_TSQR_HPP

// includes
#include <limits>
#include <numeric>
#include <Eigen/Dense>
#include "pitts_tensortrain.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate tensor-train decomposition of a tensor stored in fully dense format
  //!
  //! @tparam T         underlying data type (double, complex, ...)
  //!
  //! @param X              input tensor, modified / destroyed on output, dimension must be (size/lastDim, lastDim) where lastDim = dimensions.back()
  //! @param dimensions     tensor dimensions, input is interpreted in Fortran storage order (first index changes the fastest)
  //! @param rankTolerance  approximation accuracy, used to reduce the TTranks of the resulting tensor train
  //! @param maxRank        maximal TTrank (bond dimension), unbounded by default
  //! @return               resulting tensor train
  //!
  template<typename T>
  TensorTrain<T> fromDense_TSQR(MultiVector<T>&& X, const std::vector<int>& dimensions, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = -1)
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
    const auto nDims = dimensions.size();
    if( X.rows() != totalSize/dimensions[nDims-1] || X.cols() != dimensions[nDims-1] )
      throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::fromDense");

    TensorTrain<T> result(dimensions);

    // actually convert to tensor train format
    Tensor2<T> tmpR;
    MultiVector<T> buff;
    using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    //Eigen::BDCSVD<EigenMatrix> svd;
    for(int iDim = nDims-1; iDim > 0; iDim--)
    {
std::cout << "iDim: " << iDim << ", matrix dimensions: " << X.rows() << " x " << X.cols() << "\n";
      if( X.rows() > 10*X.cols())
      {
        // calculate QR decomposition
        block_TSQR(X, tmpR);
      }
      else
      {
        // not tall and skinny, just copy to tmpR for simplicity
        tmpR.resize(X.rows(), X.cols());
        for(int j = 0; j < X.cols(); j++)
          for(int i = 0; i < X.rows(); i++)
            tmpR(i,j) = X(i,j);
      }
//std::cout << "tmpR:\n" << ConstEigenMap(tmpR) << "\n";

      // calculate SVD of R
      //svd.compute(ConstEigenMap(tmpR), Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::JacobiSVD<EigenMatrix> svd(ConstEigenMap(tmpR), Eigen::ComputeThinU | Eigen::ComputeThinV);
      svd.setThreshold(rankTolerance);
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
      transform(X, tmpR, buff, {X.rows()/nextDim, rank*nextDim});
      std::swap(X, buff);
    }
    // last sub-tensor is now in X
    auto& lastSubT = result.editableSubTensors()[0];
    lastSubT.resize(1, dimensions[0], X.cols()/dimensions[0]);
    for(int i = 0; i < lastSubT.n(); i++)
      for(int j = 0; j < lastSubT.r2(); j++)
        lastSubT(0, i, j) = X(0, i+j*dimensions[0]);

    return result;
  }

}


#endif // PITTS_TENSORTRAIN_FROM_DENSE_TSQR_HPP
