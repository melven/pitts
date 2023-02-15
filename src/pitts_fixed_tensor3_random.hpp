/*! @file pitts_fixed_tensor3_random.hpp
* @brief fill simple fixed-dimension rank-3 tensor with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-28
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_tensor3_random.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_FIXED_TENSOR3_RANDOM)
import pitts_fixed_tensor3_random;
#define PITTS_FIXED_TENSOR3_RANDOM_HPP
#endif

// include guard
#ifndef PITTS_FIXED_TENSOR3_RANDOM_HPP
#define PITTS_FIXED_TENSOR3_RANDOM_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#ifdef PITTS_USE_MODULES
// workaround for mismatching std::align implementation
#include <memory>
#endif

#include <random>
#include "pitts_fixed_tensor3.hpp"
#include "pitts_performance.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_fixed_tensor3_random;
# define PITTS_MODULE_EXPORT export
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
{
  //! fill a rank-3 tensor with random values
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  dimension
  //!
  template<typename T, int N>
  void randomize(FixedTensor3<T,N>& t3)
  {
    const auto r1 = t3.r1();
    const auto r2 = t3.r2();
    const auto n = t3.n();

    const auto timer = PITTS::performance::createScopedTimer<FixedTensor3<T,N>>(
        {{"r1", "r2"}, {r1, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>()}}  // data
        );

    std::random_device randomSeed;
    std::mt19937 randomGenerator(randomSeed());
    std::uniform_real_distribution<T> distribution(T(-1), T(1));

    for(int i = 0; i < r1; i++)
      for(int j = 0; j < n; j++)
        for(int k = 0; k < r2; k++)
          t3(i,j,k) = distribution(randomGenerator);
  }

  //! fill a rank-3 tensor with random values (specialization for complex numbers)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  dimension
  //!
  template<typename T, int N>
  void randomize(FixedTensor3<std::complex<T>,N>& t3)
  {
    const auto r1 = t3.r1();
    const auto r2 = t3.r2();
    const auto n = t3.n();

    const auto timer = PITTS::performance::createScopedTimer<FixedTensor3<T,N>>(
        {{"r1", "r2"}, {r1, r2}},   // arguments
        {{r1*n*r2*kernel_info::NoOp<T>()},    // flops
         {r1*n*r2*kernel_info::Store<T>()}}  // data
        );

    std::random_device randomSeed;
    std::mt19937 randomGenerator(randomSeed());
    std::uniform_real_distribution<T> distribution(T(-1), T(1));

    for(int i = 0; i < r1; i++)
      for(int j = 0; j < n; j++)
        for(int k = 0; k < r2; k++)
        {
          // generate uniformly distributed numbers inside the unit circle in the complex plane
          T tmp_r, tmp_i;
          do
          {
            tmp_r = distribution(randomGenerator);
            tmp_i = distribution(randomGenerator);
          }while( tmp_r*tmp_r + tmp_i*tmp_i > 1 );

          t3(i,j,k).real(tmp_r);
          t3(i,j,k).imag(tmp_i);
        }
  }

  // explicit template instantiations
}


#endif // PITTS_FIXED_TENSOR3_RANDOM_HPP
