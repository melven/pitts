/*! @file pitts_multivector_transpose.hpp
* @brief reshape and rearrange entries in a tall-skinny matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-08-06
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TRANSPOSE_HPP
#define PITTS_MULTIVECTOR_TRANSPOSE_HPP

// includes
#include "pitts_multivector.hpp"
#include <array>
#include <tuple>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! reshape and transpose a tall-skinny matrix
  //!
  //! This is equivalent to first adjusting the shape (but keeping the data in the same ordering) and then transposing
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param Y        output mult-vector, resized to dimensions (k, l) or desired shape (see below)
  //! @param reshape  desired new shape (k, l) where l is large and k is small
  //! @param reverse  first transpose, then reshape instead (changes the ordering)
  //!
  template<typename T>
  void transpose(const MultiVector<T>& X, MultiVector<T>& Y, std::array<long long,2> reshape = {0, 0}, bool reverse = false);
  
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_transpose_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_TRANSPOSE_HPP
