/*! @file pitts_tensor3_random.hpp
* @brief fill simple rank-3 tensor with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR3_RANDOM_HPP
#define PITTS_TENSOR3_RANDOM_HPP

// includes
#include "pitts_tensor3.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! fill a rank-3 tensor with random values
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  void randomize(Tensor3<T>& t3);
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensor3_random_impl.hpp"
#endif

#endif // PITTS_TENSOR3_RANDOM_HPP
