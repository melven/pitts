/*! @file pitts_multivector_random_impl.hpp
* @brief fill multivector (simple rank-2 tensor) with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_RANDOM_IMPL_HPP
#define PITTS_MULTIVECTOR_RANDOM_IMPL_HPP

// includes
#include <random>
#include <complex>
#include "pitts_multivector_random.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! fill a multivector (rank-2 tensor) with random values
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  void randomize(MultiVector<T>& X)
  {
    const auto rows = X.rows();
    const auto cols = X.cols();

    // gather performance data
    const double rowsd = rows, colsd = cols;
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"rows", "cols"},{rows, cols}}, // arguments
        {{rowsd*colsd*kernel_info::NoOp<T>()}, // flops
         {rowsd*colsd*kernel_info::Store<T>()}} // data transfers
        );

    std::random_device randomSeed;

#pragma omp parallel
    {
      std::random_device::result_type seed;
#pragma omp critical(PITTS_MULTIVECTOR_RANDOM)
      {
        seed = randomSeed();
      }
      std::mt19937 randomGenerator(seed);
      using RealType = decltype(std::abs(T(0)));
      std::uniform_real_distribution<RealType> distribution(RealType(-1), RealType(1));
      for(long long j = 0; j < cols; j++)
      {
#pragma omp for schedule(static) nowait
        for(long long i = 0; i < rows; i++)
          X(i,j) = distribution(randomGenerator);
      }
    }
  }

  //! fill a multivector (rank-2 tensor) with random values (specialization for std::complex)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  void randomize(MultiVector<std::complex<T>>& X)
  {
    const auto rows = X.rows();
    const auto cols = X.cols();

    // gather performance data
    const double rowsd = rows, colsd = cols;
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"rows", "cols"},{rows, cols}}, // arguments
        {{rowsd*colsd*kernel_info::NoOp<std::complex<T>>()}, // flops
         {rowsd*colsd*kernel_info::Store<std::complex<T>>()}} // data transfers
        );

    std::random_device randomSeed;

#pragma omp parallel
    {
      std::random_device::result_type seed;
#pragma omp critical(PITTS_MULTIVECTOR_RANDOM)
      {
        seed = randomSeed();
      }
      std::mt19937 randomGenerator(seed);
      std::uniform_real_distribution<T> distribution(T(-1), T(1));
      for(long long j = 0; j < cols; j++)
      {
#pragma omp for schedule(static) nowait
        for(long long i = 0; i < rows; i++)
        {
          // generate uniformly distributed numbers inside the unit circle in the complex plane
          T tmp_r, tmp_i;
          do
          {
            tmp_r = distribution(randomGenerator);
            tmp_i = distribution(randomGenerator);
          }while( tmp_r*tmp_r + tmp_i*tmp_i > 1 );

          X(i,j).real(tmp_r);
          X(i,j).imag(tmp_i);
        }
      }
    }
  }
}


#endif // PITTS_MULTIVECTOR_RANDOM_IMPL_HPP
