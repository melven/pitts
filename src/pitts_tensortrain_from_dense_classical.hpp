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
#include <limits>
#include <cmath>
#include <vector>
#include "pitts_tensortrain.hpp"

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
  TensorTrain<T> fromDense_classical(const Iter first, const Iter last, const std::vector<int>& dimensions, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = -1);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_from_dense_classical_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_FROM_DENSE_CLASSICAL_HPP
