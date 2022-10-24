/*! @file pitts_tensortrain_from_dense_classical.hpp
* @brief conversion of a dense tensor to the tensor-train format, classical TT-SVD algorithm
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_FROM_DENSE_CLASSICAL_HPP
#define PITTS_TENSORTRAIN_FROM_DENSE_CLASSICAL_HPP

// includes
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options
#include "pitts_tensortrain.hpp"
#include "pitts_timer.hpp"
#include "pitts_tensor3_fold.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate tensor-train decomposition of a tensor stored in fully dense format (slow classical TT-SVD)
  //!
  //! @tparam Iter      contiguous input iterator to access the dense data
  //! @tparam T         underlying data type (double, complex, ...)
  //!
  //! @param first          input iterator that points to the first index, e.g. std::begin(someContainer)
  //! @param last           input iterator that points behind the last index, e.g. std::end(someContainer)
  //! @param dimensions     tensor dimensions, input is interpreted in Fortran storage order (first index changes the fastest)
  //! @param rankTolerance  approximation accuracy, used to reduce the TTranks of the resulting tensor train
  //! @param maxRank        maximal TTrank (bond dimension), unbounded by default
  //! @return               resulting tensor train
  //!
  template<class Iter, typename T = std::iterator_traits<Iter>::value_type>
  TensorTrain<T> fromDense_classical(const Iter first, const Iter last, const std::vector<int>& dimensions, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = -1)
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

    std::vector<Tensor3<T>> subTensors(dimensions.size());
    using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    EigenMatrix tmp = Eigen::Map<const EigenMatrix>(&(*first), 1, totalSize);
    for(int iDim = 0; iDim+1 < dimensions.size(); iDim++)
    {
      tmp.resize(tmp.rows()*dimensions[iDim], tmp.cols()/dimensions[iDim]);
      Eigen::BDCSVD<EigenMatrix> svd(tmp, Eigen::ComputeThinU | Eigen::ComputeThinV);
      svd.setThreshold(rankTolerance);
      int rank = svd.rank();
      if( maxRank > 0 )
        rank = std::min(rank, maxRank);

      fold_left(svd.matrixU().leftCols(rank), dimensions[iDim], subTensors[iDim]);

      tmp.resize(rank, tmp.cols());
      tmp = svd.singularValues().topRows(rank).asDiagonal() * svd.matrixV().leftCols(rank).adjoint();
    }
    int lastDim = dimensions.size()-1;
    tmp.resize(tmp.rows()*dimensions[lastDim], tmp.cols()/dimensions[lastDim]);
    fold_left(tmp, dimensions[lastDim], subTensors[lastDim]);

    TensorTrain<T> result(dimensions);

    result.setSubTensors(0, std::move(subTensors));

    return result;
  }

}


#endif // PITTS_TENSORTRAIN_FROM_DENSE_CLASSICAL_HPP
