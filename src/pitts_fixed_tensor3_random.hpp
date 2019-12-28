/*! @file pitts_fixed_tensor3_random.hpp
* @brief fill simple fixed-dimension rank-3 tensor with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-28
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_FIXED_TENSOR3_RANDOM_HPP
#define PITTS_FIXED_TENSOR3_RANDOM_HPP

// includes
#include <random>
#include "pitts_fixed_tensor3.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! fill a rank-3 tensor with random values
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  dimension
  //!
  template<typename T, int N>
  void randomize(FixedTensor3<T,N>& t3)
  {
    std::random_device randomSeed;
    std::mt19937 randomGenerator(randomSeed());
    std::uniform_real_distribution<T> distribution(T(-1), T(1));

    const auto r1 = t3.r1();
    const auto r2 = t3.r2();
    const auto n = t3.n();
    for(int i = 0; i < r1; i++)
      for(int j = 0; j < n; j++)
        for(int k = 0; k < r2; k++)
          t3(i,j,k) = distribution(randomGenerator);
  }

}


#endif // PITTS_FIXED_TENSOR3_RANDOM_HPP
