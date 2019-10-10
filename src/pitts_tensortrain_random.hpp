/*! @file pitts_tensortrain_random.hpp
* @brief fill simple tensor train with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_RANDOM_HPP
#define PITTS_TENSORTRAIN_NORM_HPP

// includes
#include <random>
#include "pitts_tensortrain.hpp"

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
    std::random_device randomSeed;
    std::mt19937 randomGenerator(randomSeed());
    std::uniform_real_distribution<T> distribution(T(-1), T(1));
    for(auto& subT: TT.editableSubTensors())
    {
      const auto r1 = subT.r1();
      const auto r2 = subT.r2();
      const auto n = subT.n();
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2; k++)
            subT(i,j,k) = distribution(randomGenerator);
    }
  }

}


#endif // PITTS_TENSORTRAIN_RANDOM_HPP
