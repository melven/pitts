/*! @file pitts_fixed_tensortrain_random.hpp
* @brief fill simple fixed-dimension tensor train with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-28
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_fixed_tensortrain_random.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_FIXED_TENSORTRAIN_RANDOM)
import pitts_fixed_tensortrain_random;
#define PITTS_FIXED_TENSORTRAIN_RANDOM_HPP
#endif

// include guard
#ifndef PITTS_FIXED_TENSORTRAIN_RANDOM_HPP
#define PITTS_FIXED_TENSORTRAIN_RANDOM_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include "pitts_fixed_tensortrain.hpp"
#include "pitts_fixed_tensor3_random.hpp"
#include "pitts_timer.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_fixed_tensortrain_random;
# define PITTS_MODULE_EXPORT export
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
{
  //! fill a tensor train format with random values (keeping current TT-ranks)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  dimensions
  //!
  template<typename T, int N>
  void randomize(FixedTensorTrain<T,N>& TT)
  {
    const auto timer = PITTS::timing::createScopedTimer<FixedTensorTrain<T,N>>();

    for(auto& subT: TT.editableSubTensors())
      randomize(subT);
  }

  // explicit template instantiations
}


#endif // PITTS_FIXED_TENSORTRAIN_RANDOM_HPP
