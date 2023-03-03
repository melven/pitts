/*! @file pitts_multivector_random.hpp
* @brief fill multivector (simple rank-2 tensor) with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_RANDOM_HPP
#define PITTS_MULTIVECTOR_RANDOM_HPP

// includes
#include "pitts_multivector.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! fill a multivector (rank-2 tensor) with random values
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  void randomize(MultiVector<T>& X);
  
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_random_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_RANDOM_HPP
