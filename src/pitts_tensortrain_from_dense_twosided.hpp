/*! @file pitts_tensortrain_from_dense_twosided.hpp
* @brief conversion of a dense tensor to the tensor-train format (based on a hopefully faster TSQR algorithm, two-sided variant)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-08-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_HPP
#define PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_HPP

// includes
#include <limits>
#include <numeric>
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options
#include "pitts_tensortrain.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_transpose.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
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
      void split(const MultiVector<T>& X, MultiVector<T>& Y, Tensor2<T>& M, int nextDim, T rankTolerance, int maxRank)
      {
        using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        Eigen::JacobiSVD<EigenMatrix> svd;
std::cout << "HOSVD::split  matrix dimensions: " << X.rows() << " x " << X.cols() << "\n";
        if( X.rows() > 10*X.cols() )
        {
          // calculate QR decomposition (QR-trick: X=QR,SVD(R))
          block_TSQR(X, M, 0, false);
          svd.compute(ConstEigenMap(M), Eigen::ComputeThinU | Eigen::ComputeThinV);
        }
        else
        {
          svd.compute(ConstEigenMap(X), Eigen::ComputeThinU | Eigen::ComputeThinV);
        }

std::cout << "singular values: " << svd.singularValues().transpose() << "\n";

        // truncate svd
        svd.setThreshold(rankTolerance);
        int rank = svd.rank();
        if( maxRank > 0 )
          rank = std::min(maxRank, rank);

        // copy right singular vectors
        M.resize(X.cols(), rank);
        EigenMap(M) = svd.matrixV().leftCols(rank);

        // transform input, s.t. Y \approx X M
        transform(X, M, Y, {X.rows()/nextDim, rank*nextDim});
      }
    }
  }


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
  //! @return               resulting tensor train
  //!
  template<typename T>
  TensorTrain<T> fromDense_twoSided(MultiVector<T>& X, MultiVector<T>& work, const std::vector<int>& dimensions, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = -1)
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
    Tensor2<T> M;
    for(int ii = 0; ii < nDims; ii++)
    {
      if( ii % 2 == 0 )
      {
        // right part
        const auto iDim = nDims - 1 - ii/2;
        if( ii > 0 )
        {
          const auto r1 = result.subTensors()[iDim+1].r1();
          const auto n = dimensions[iDim];
          transpose(work, X, {(work.rows()*work.cols())/(n*r1), n*r1}, true);
        }
        if( ii != nDims-1 )
        {
          internal::HOSVD::split(X, work, M, 1, rankTolerance, maxRank);
        }
        else
        {
          M.resize(X.cols(), X.rows());
          EigenMap(M) = ConstEigenMap(X).transpose();
        }

        auto& subT = result.editableSubTensors()[iDim];
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
          const auto r2 = (iDim == 0 ) ? 1 : result.subTensors()[iDim-1].r2();
          const auto n = dimensions[iDim];
          transpose(work, X, {(work.cols()*work.rows())/(n*r2), n*r2}, false);
        }
        if( ii != nDims - 1 )
        {
          internal::HOSVD::split(X, work, M, 1, rankTolerance, maxRank);
        }
        else
        {
          M.resize(X.cols(), X.rows());
          EigenMap(M) = ConstEigenMap(X).transpose();
        }

        auto& subT = result.editableSubTensors()[iDim];
        const int rank = M.r2();
        subT.resize(X.cols()/dimensions[iDim], dimensions[iDim], rank);
        for(int i = 0; i < rank; i++)
          for(int j = 0; j < subT.n(); j++)
            for(int k = 0; k < subT.r1(); k++)
              subT(k,j,i) = M(k+j*subT.r1(), i);
      }
    }

    return result;
  }

}


#endif // PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_HPP
