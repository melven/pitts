// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_norm_impl.hpp
* @brief calculate the 2-norm each vector in a multi-vector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_NORM_IMPL_HPP
#define PITTS_MULTIVECTOR_NORM_IMPL_HPP

// includes
#include <stdexcept>
#include "pitts_multivector_norm.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement multivector norm
  template<typename T>
  Eigen::ArrayX<T> norm2(const MultiVector<T>& X)
  {
    const auto nChunks = X.rowChunks();
    const auto nCols = X.cols();

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"nChunks", "nCols"},{nChunks, nCols}}, // arguments
        {{nCols*nChunks*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {nChunks*nCols*kernel_info::Load<Chunk<T>>() + nCols*kernel_info::Store<T>()}} // data transfers
        );

    T tmp[nCols];
    if( nCols == 1 )
    {
      // special case because the OpenMP vector reduction has significant overhead (factor 10)
      T beta = T(0);
#pragma omp parallel reduction(+:beta) if(nChunks > 50)
      {
        auto [iThread,nThreads] = internal::parallel::ompThreadInfo();

        constexpr auto unroll = 5;
        const auto nIter = nChunks / unroll;

        const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});

        const Chunk<T>* pX = &X.chunk(0, 0);
        Chunk<T> tmpChunk[unroll];
        for(int i = 0; i < unroll; i++)
          tmpChunk[i] = Chunk<T>{};

        for(long long iter = firstIter; iter <= lastIter; iter++)
          for(int i = 0; i < unroll; i++)
            fmadd(pX[iter*unroll+i], pX[iter*unroll+i], tmpChunk[i]);

        if( iThread == nThreads - 1 )
          for(long long iChunk = nIter*unroll; iChunk < nChunks; iChunk++)
            fmadd(pX[iChunk], pX[iChunk], tmpChunk[iChunk-nIter*unroll]);

        for(int i = 0; i < unroll; i++)
          beta += sum(tmpChunk[i]);
      }
      tmp[0] = beta;
    }
    else
    {
      for(int iCol = 0; iCol < nCols; iCol++)
        tmp[iCol] = T(0);
#pragma omp parallel reduction(+:tmp) if(nChunks > 50)
      {
        auto [iThread,nThreads] = internal::parallel::ompThreadInfo();

        constexpr auto unroll = 5;
        const auto nIter = nChunks / unroll;

        const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});

        for(int iCol = 0; iCol < nCols; iCol++)
        {
          const Chunk<T>* pX = &X.chunk(0, iCol);
          Chunk<T> tmpChunk[unroll];
          for(int i = 0; i < unroll; i++)
            tmpChunk[i] = Chunk<T>{};

          for(long long iter = firstIter; iter <= lastIter; iter++)
            for(int i = 0; i < unroll; i++)
              fmadd(pX[iter*unroll+i], pX[iter*unroll+i], tmpChunk[i]);

          if( iThread == nThreads - 1 )
            for(long long iChunk = nIter*unroll; iChunk < nChunks; iChunk++)
              fmadd(pX[iChunk], pX[iChunk], tmpChunk[iChunk-nIter*unroll]);

          tmp[iCol] = T{};
          for(int i = 0; i < unroll; i++)
            tmp[iCol] += sum(tmpChunk[i]);
        }
      }
    }

    using arr = Eigen::ArrayX<T>;
    arr result = Eigen::Map<arr>(tmp, nCols).sqrt();
    return result;
  }

}


#endif // PITTS_MULTIVECTOR_NORM_IMPL_HPP
