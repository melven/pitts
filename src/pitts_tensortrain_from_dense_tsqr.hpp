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
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <Eigen/Dense>
#include "pitts_tensortrain.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate tensor-train decomposition of a tensor stored in fully dense format
  //!
  //! @tparam Iter      contiguous input iterator to access the dense data
  //! @tparam T         underlying data type (double, complex, ...)
  //!
  //! @param first          input iterator that points to the first index, e.g. std::begin(someContainer)
  //! @param last           input iterator that points behind the last index, e.g. std::end(someContainer)
  //! @param dimensions     tensor dimensions, input is interpreted in Fortran storage order (first index changes the fastest)
  //! @param rankTolerance  approximation accuracy, used to reduce the TTranks of the resulting tensor train
  //! @return               resulting tensor train
  //!
  template<class Iter, typename T = std::iterator_traits<Iter>::value_type>
  TensorTrain<T> fromDense_TSQR(const Iter first, const Iter last, const std::vector<int>& dimensions, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()))
  {
    // timer
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // check that the input is contiguous in memory
    //static_assert(std::is_base_of< std::contiguous_iterator_tag, typename std::iterator_traits<Iter>::iterator_category >::value, "fromDense only works with contiguous iterators!");
    static_assert(std::is_base_of< std::random_access_iterator_tag, typename std::iterator_traits<Iter>::iterator_category >::value, "fromDense only works with contiguous iterators!");

    // abort early for zero dimensions
    if( dimensions.size() == 0 )
    {
      if( last - first != 0 )
        throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::fromDense");
      return TensorTrain<T>{dimensions};
    }

    const auto totalSize = std::accumulate(begin(dimensions), end(dimensions), (std::ptrdiff_t)1, std::multiplies<std::ptrdiff_t>());
    if( totalSize != last - first )
      throw std::out_of_range("Mismatching dimensions in TensorTrain<T>::fromDense");
    const auto nDims = dimensions.size();

    TensorTrain<T> result(dimensions);

    // copy to padded buffer
    MultiVector<T> tmp(totalSize/dimensions[nDims-1], dimensions[nDims-1]);
#pragma omp parallel for schedule(static)
    for(int iChunk = 0; iChunk < tmp.rowChunks(); iChunk++)
    {
      for(int j = 0; j < tmp.cols(); j++)
      {
        for(int ii = 0; ii < Chunk<T>::size; ii++)
        {
          int i = iChunk*Chunk<T>::size + ii;
          if( i >= tmp.rows() )
            continue;
          tmp.chunk(iChunk,j)[ii] = first[i+j*tmp.rows()];
        }
      }
    }

    // actually convert to tensor train format
    Tensor2<T> tmpR;
    MultiVector<T> buff;
    using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    //Eigen::BDCSVD<EigenMatrix> svd;
    for(int iDim = nDims-1; iDim > 0; iDim--)
    {
std::cout << "iDim: " << iDim << ", matrix dimensions: " << tmp.rows() << " x " << tmp.cols() << "\n";
      if( tmp.rows() > 10*tmp.cols())
      {
        // calculate QR decomposition
        block_TSQR(tmp, tmpR);
      }
      else
      {
        // not tall and skinny, just copy to tmpR for simplicity
        tmpR.resize(tmp.rows(), tmp.cols());
        for(int j = 0; j < tmp.cols(); j++)
          for(int i = 0; i < tmp.rows(); i++)
            tmpR(i,j) = tmp(i,j);
      }
//std::cout << "tmpR:\n" << ConstEigenMap(tmpR) << "\n";

      // calculate SVD of R
      //svd.compute(ConstEigenMap(tmpR), Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::JacobiSVD<EigenMatrix> svd(ConstEigenMap(tmpR), Eigen::ComputeThinU | Eigen::ComputeThinV);
      svd.setThreshold(rankTolerance);
std::cout << "singular values: " << svd.singularValues().transpose() << "\n";

      // copy V to the TT sub-tensor
      svd.setThreshold(rankTolerance);
      const int rank = svd.rank();
      auto& subT = result.editableSubTensors()[iDim];
      subT.resize(rank, dimensions[iDim], tmp.cols()/dimensions[iDim]);
      for(int i = 0; i < subT.r1(); i++)
        for(int j = 0; j < subT.n(); j++)
          for(int k = 0; k < subT.r2(); k++)
            subT(i,j,k) = svd.matrixV()(j+subT.n()*k, i);

      // update tmp <- tmp * V
      buff.resize(tmp.rows(), rank);
#pragma omp parallel for schedule(static)
      for(int iChunk = 0; iChunk < tmp.rowChunks(); iChunk++)
      {
        for(int j = 0; j < rank; j++)
        {
          buff.chunk(iChunk,j) = Chunk<T>{};
          for(int k = 0; k < tmp.cols(); k++)
            fmadd(svd.matrixV()(k,j), tmp.chunk(iChunk,k), buff.chunk(iChunk,j));
        }
      }

      const auto nextDim = dimensions[iDim-1];
      tmp.resize(buff.rows()/nextDim, buff.cols()*nextDim);
#pragma omp parallel for schedule(static)
      for(int iChunk = 0; iChunk < tmp.rowChunks(); iChunk++)
      {
        for(int j = 0; j < tmp.cols(); j++)
        {
          for(int ii = 0; ii < Chunk<T>::size; ii++)
          {
            int i = iChunk*Chunk<T>::size + ii;
            if( i >= tmp.rows() )
              continue;
            tmp.chunk(iChunk,j)[ii] = buff(i+(j%nextDim)*tmp.rows(),j/nextDim);
          }
        }
      }

    }
    // last sub-tensor is now in tmp
    auto& lastSubT = result.editableSubTensors()[0];
    lastSubT.resize(1, dimensions[0], tmp.cols()/dimensions[0]);
    for(int i = 0; i < lastSubT.n(); i++)
      for(int j = 0; j < lastSubT.r2(); j++)
        lastSubT(0, i, j) = tmp(0, i+j*dimensions[0]);

    return result;
  }

}


#endif // PITTS_TENSORTRAIN_FROM_DENSE_TSQR_HPP
