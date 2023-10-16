// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_random_impl.hpp
* @brief fill multivector (simple rank-2 tensor) with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-10
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_RANDOM_IMPL_HPP
#define PITTS_MULTIVECTOR_RANDOM_IMPL_HPP

// includes
#include "pitts_multivector_random.hpp"
#include "pitts_parallel.hpp"
#include "pitts_random.hpp"
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
    const double rowsd = rows;
    const double colsd = cols;
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"rows", "cols"},{rows, cols}}, // arguments
        {{rowsd*colsd*kernel_info::NoOp<T>()}, // flops
         {rowsd*colsd*kernel_info::Store<T>()}} // data transfers
        );

    // not using non-temporal stores so far, but for now this is not performance-critical

    // save the state of the random number generator to avoid a race condition
    const auto randomGeneratorState = internal::randomGenerator;
#pragma omp parallel
    {
      const auto& [iThread, nThreads] = internal::parallel::ompThreadInfo();
      const auto& [firstElem, lastElem] = internal::parallel::distribute(rows, {iThread, nThreads});

      internal::UniformUnitDistribution<T> distribution;
      auto randomGenerator = randomGeneratorState;
      distribution.discard(firstElem*cols, randomGenerator);

      for(long long i = firstElem; i <= lastElem; i++)
      {
        for(long long j = 0; j < cols; j++)
          X(i,j) = distribution(randomGenerator);
      }

      // ensure the global randomGenerator is advanced correctly
      if( iThread == nThreads-1 )
        internal::randomGenerator = randomGenerator;
    }
  }
}


#endif // PITTS_MULTIVECTOR_RANDOM_IMPL_HPP
