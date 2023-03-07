/*! @file pitts_tensortrain_axpby_plain.hpp
* @brief addition for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-11-07
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_AXPBY_PLAIN_HPP
#define PITTS_TENSORTRAIN_AXPBY_PLAIN_HPP

// includes
#include <limits>
#include <cmath>
#include "pitts_tensortrain.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    // implement plain variant for TT axpby
    template<typename T>
    T axpby_plain(T alpha, const TensorTrain<T>& TTx, T beta, TensorTrain<T>& TTy, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max());

  } // namespace internal

} // namespace pitts

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_axpby_plain_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_AXPBY_PLAIN_HPP
