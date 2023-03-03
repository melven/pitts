/*! @file pitts_tensortrain_impl.hpp
* @brief simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_IMPL_HPP
#define PITTS_TENSORTRAIN_IMPL_HPP

// includes
#include "pitts_tensortrain.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement TT copy
  template<typename T>
  void copy(const TensorTrain<T>& a, TensorTrain<T>& b)
  {
    // check that dimensions match
    if( a.dimensions() != b.dimensions() )
      throw std::invalid_argument("TensorTrain copy dimension mismatch!");
    
    b.setTTranks(a.getTTranks());

    for(int i = 0; i < a.dimensions().size(); i++)
    {
      const auto& t3a = a.subTensor(i);
      const auto copyFcn = [&t3a](Tensor3<T>& t3b){copy(t3a, t3b);};
      b.editSubTensor(i, copyFcn, a.isOrthonormal(i));
    }
  }
}


#endif // PITTS_TENSORTRAIN_IMPL_HPP
