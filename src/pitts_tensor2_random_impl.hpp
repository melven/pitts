/*! @file pitts_tensor2_random_impl.hpp
* @brief fill simple rank-2 tensor with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR2_RANDOM_IMPL_HPP
#define PITTS_TENSOR2_RANDOM_IMPL_HPP

// includes
#include "pitts_tensor2_random.hpp"
#include "pitts_random.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement tensor2 random
  template<typename T>
  void randomize(Tensor2<T>& t2)
  {
    const auto r1 = t2.r1();
    const auto r2 = t2.r2();
    const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"r1", "r2"}, {r1, r2}},   // arguments
        {{r1*r2*kernel_info::NoOp<T>()},    // flops
         {r1*r2*kernel_info::Store<T>()}}  // data
        );
    
    internal::UniformUnitDistribution<T> distribution;

    for(long long i = 0; i < r1; i++)
      for(long long j = 0; j < r2; j++)
          t2(i,j) = distribution(internal::randomGenerator);
  }

}


#endif // PITTS_TENSOR2_RANDOM_IMPL_HPP
