/*! @file pitts_tensor3_random_impl.hpp
* @brief fill simple rank-3 tensor with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR3_RANDOM_IMPL_HPP
#define PITTS_TENSOR3_RANDOM_IMPL_HPP

// includes
#include "pitts_tensor3_random.hpp"
#include "pitts_random.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement tensor3 random
  template<typename T>
  void randomize(Tensor3<T>& t3)
  {
    const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

    internal::UniformUnitDistribution<T> distribution;

    const auto r1 = t3.r1();
    const auto r2 = t3.r2();
    const auto n = t3.n();
    for(int i = 0; i < r1; i++)
      for(int j = 0; j < n; j++)
        for(int k = 0; k < r2; k++)
          t3(i,j,k) = distribution(internal::randomGenerator);
  }

}


#endif // PITTS_TENSOR3_RANDOM_IMPL_HPP
