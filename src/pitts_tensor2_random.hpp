/*! @file pitts_tensor2_random.hpp
* @brief fill simple rank-2 tensor with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR2_RANDOM_HPP
#define PITTS_TENSOR2_RANDOM_HPP

// includes
#include <random>
#include "pitts_tensor2.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! fill a rank-2 tensor with random values
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
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


    std::random_device randomSeed;
    std::mt19937 randomGenerator(randomSeed());
    std::uniform_real_distribution<T> distribution(T(-1), T(1));

    for(long long i = 0; i < r1; i++)
      for(long long j = 0; j < r2; j++)
          t2(i,j) = distribution(randomGenerator);
  }

}


#endif // PITTS_TENSOR2_RANDOM_HPP
