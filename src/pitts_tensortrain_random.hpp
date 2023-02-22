/*! @file pitts_tensortrain_random.hpp
* @brief fill simple tensor train with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_RANDOM_HPP
#define PITTS_TENSORTRAIN_RANDOM_HPP

// includes
#include "pitts_tensortrain.hpp"
#include "pitts_tensor3_random.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
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

}


#endif // PITTS_TENSORTRAIN_RANDOM_HPP
