/*! @file pitts_tensortrain_random.hpp
* @brief fill simple tensor train with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_tensortrain_random.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_TENSORTRAIN_RANDOM)
import pitts_tensortrain_random;
#define PITTS_TENSORTRAIN_RANDOM_HPP
#endif

// include guard
#ifndef PITTS_TENSORTRAIN_RANDOM_HPP
#define PITTS_TENSORTRAIN_RANDOM_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include "pitts_tensortrain.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensor3_random.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_tensortrain_random;
# define PITTS_MODULE_EXPORT export
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
{
  //! fill a tensor train format with random values (keeping current TT-ranks)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  void randomize(TensorTrain<T>& TT)
  {
    for(int iDim = 0; iDim < TT.dimensions().size(); iDim++)
    {
      constexpr auto randomizeFcn = [](Tensor3<T>& subT) {randomize(subT);};
      TT.editSubTensor(iDim, randomizeFcn, TT_Orthogonality::none);
    }
  }

  // explicit template instantiations
  template void randomize<float>(TensorTrain<float>& TT);
  template void randomize<double>(TensorTrain<double>& TT);
}


#endif // PITTS_TENSORTRAIN_RANDOM_HPP
