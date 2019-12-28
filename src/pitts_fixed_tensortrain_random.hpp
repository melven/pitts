/*! @file pitts_fixed_tensortrain_random.hpp
* @brief fill simple fixed-dimension tensor train with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-28
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_FIXED_TENSORTRAIN_RANDOM_HPP
#define PITTS_FIXED_TENSORTRAIN_RANDOM_HPP

// includes
#include <random>
#include "pitts_fixed_tensortrain.hpp"
#include "pitts_fixed_tensor3_random.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! fill a tensor train format with random values (keeping current TT-ranks)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  dimensions
  //!
  template<typename T, int N>
  void randomize(FixedTensorTrain<T,N>& TT)
  {
    for(auto& subT: TT.editableSubTensors())
      randomize(subT);
  }

}


#endif // PITTS_FIXED_TENSORTRAIN_RANDOM_HPP
