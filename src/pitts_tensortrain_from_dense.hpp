/*! @file pitts_tensortrain_from_dense.hpp
* @brief conversion of a dense tensor to the tensor-train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_FROM_DENSE_HPP
#define PITTS_TENSORTRAIN_FROM_DENSE_HPP

// includes
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <Eigen/Dense>
#include "pitts_tensortrain.hpp"
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
  TensorTrain<T> fromDense(Iter first, Iter last, const std::vector<int>& dimensions, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()))
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

    TensorTrain<T> result(dimensions);

    using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    EigenMatrix tmp = Eigen::Map<const EigenMatrix>(&(*first), 1, totalSize);
    for(int iDim = 0; iDim+1 < dimensions.size(); iDim++)
    {
      tmp.resize(tmp.rows()*dimensions[iDim], tmp.cols()/dimensions[iDim]);
      Eigen::BDCSVD<EigenMatrix> svd(tmp, Eigen::ComputeThinU | Eigen::ComputeThinV);
      svd.setThreshold(rankTolerance);
      const int rank = svd.rank();

      auto& subT = result.editableSubTensors()[iDim];
      subT.resize(tmp.rows()/dimensions[iDim], dimensions[iDim], rank);
      for(int i = 0; i < subT.r1(); i++)
        for(int j = 0; j < subT.n(); j++)
          for(int k = 0; k < subT.r2(); k++)
            subT(i,j,k) = svd.matrixU()(i+j*subT.r1(), k);

      tmp.resize(rank, tmp.cols());
      tmp = svd.singularValues().topRows(rank).asDiagonal() * svd.matrixV().leftCols(rank).adjoint();
    }
    int lastDim = dimensions.size()-1;
    auto& lastSubT = result.editableSubTensors()[lastDim];
    tmp.resize(tmp.size()/dimensions[lastDim], dimensions[lastDim]);
    lastSubT.resize(tmp.rows(), tmp.cols(), 1);
    for(int j = 0; j < tmp.cols(); j++)
      for(int i = 0; i < tmp.rows(); i++)
        lastSubT(i, j, 0) = tmp(i, j);

    return result;
  }

}


#endif // PITTS_TENSORTRAIN_FROM_DENSE_HPP
