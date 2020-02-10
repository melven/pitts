/*! @file pitts_multivector_random.hpp
* @brief fill multivector (simple rank-2 tensor) with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_RANDOM_HPP
#define PITTS_MULTIVECTOR_RANDOM_HPP

// includes
#include <random>
#include "pitts_multivector.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! fill a multivector (rank-2 tensor) with random values
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  void randomize(MultiVector<T>& t2)
  {
    std::random_device randomSeed;
    std::mt19937 randomGenerator(randomSeed());
    std::uniform_real_distribution<T> distribution(T(-1), T(1));

    const auto rows = t2.rows();
    const auto cols = t2.cols();
    for(int j = 0; j < cols; j++)
      for(int i = 0; i < rows; i++)
          t2(i,j) = distribution(randomGenerator);
  }

}


#endif // PITTS_MULTIVECTOR_RANDOM_HPP
