// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_axpby_impl.hpp
* @brief calculate the scaled addition of the pair-wise columns of two multi-vectors
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_AXPBY_IMPL_HPP
#define PITTS_MULTIVECTOR_AXPBY_IMPL_HPP

// includes
#include <array>
#include <stdexcept>
#include "pitts_eigen.hpp"
#include "pitts_multivector_axpby.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_parallel.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement MultiVector axpy
  template<typename T>
  void axpy(const Eigen::ArrayX<T>& alpha, const MultiVector<T>& X, MultiVector<T>& Y)
  {
    // check dimensions
    if( X.cols() != Y.cols() || alpha.size() != X.cols() )
      throw std::invalid_argument("MultiVector::axpy: cols dimension mismatch!");

    if( X.rows() != Y.rows() )
      throw std::invalid_argument("MultiVector::axpy: rows dimension mismatch!");

    const auto nChunks = X.rowChunks();
    const auto nCols = Y.cols();

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"nChunks", "nCols"},{nChunks, nCols}}, // arguments
        {{nChunks*nCols*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {nChunks*nCols*kernel_info::Load<Chunk<T>>() + nChunks*nCols*kernel_info::Update<Chunk<T>>()}} // data transfers
        );

#pragma omp parallel if(nChunks > 50)
    {
      auto [iThread,nThreads] = internal::parallel::ompThreadInfo();

      constexpr auto unroll = 5;
      const auto nIter = nChunks / unroll;

      const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});

      for(int iCol = 0; iCol < nCols; iCol++)
      {
        const Chunk<T>* pX = &X.chunk(0, iCol);
        Chunk<T>* pY = &Y.chunk(0, iCol);
        const T tmp = alpha(iCol);

        for(long long iter = firstIter; iter <= lastIter; iter++)
        {
          for(int i = 0; i < unroll; i++)
            fmadd(tmp, pX[iter*unroll+i], pY[iter*unroll+i]);
        }

        if( iThread == nThreads - 1 )
        {
          for(long long iChunk = nIter*unroll; iChunk < nChunks; iChunk++)
            fmadd(tmp, pX[iChunk], pY[iChunk]);
        }
      }
    }
  }

  // implement MultiVector axpy+norm
  template<typename T>
  Eigen::ArrayX<T> axpy_norm2(const Eigen::ArrayX<T>& alpha, const MultiVector<T>& X, MultiVector<T>& Y)
  {
    // check dimensions
    if( X.cols() != Y.cols() || alpha.size() != X.cols() )
      throw std::invalid_argument("MultiVector::axpy_norm2: cols dimension mismatch!");

    if( X.rows() != Y.rows() )
      throw std::invalid_argument("MultiVector::axpy_norm2: rows dimension mismatch!");

    const auto nChunks = X.rowChunks();
    const auto nCols = Y.cols();

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"nChunks", "nCols"},{nChunks, nCols}}, // arguments
        {{2*nChunks*nCols*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {nChunks*nCols*kernel_info::Load<Chunk<T>>() + nChunks*nCols*kernel_info::Update<Chunk<T>>() + nCols*kernel_info::Store<T>()}} // data transfers
        );

    T tmp[nCols];
    if( nCols == 1 )
    {
      // special case because the OpenMP vector reduction has significant overhead (factor 10)
      T beta = T(0);
#pragma omp parallel reduction(+:beta) if(nChunks > 50)
      {
        auto [iThread,nThreads] = internal::parallel::ompThreadInfo();

        constexpr auto unroll = 3;
        const auto nIter = nChunks / unroll;

        const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});

        const Chunk<T>* pX = &X.chunk(0, 0);
        Chunk<T>* pY = &Y.chunk(0, 0);
        const T tmpAlpha = alpha(0);
        Chunk<T> tmpChunk[unroll];
        for(int i = 0; i < unroll; i++)
          tmpChunk[i] = Chunk<T>{};

        for(long long iter = firstIter; iter <= lastIter; iter++)
        {
          for(int i = 0; i < unroll; i++)
          {
            fmadd(tmpAlpha, pX[iter*unroll+i], pY[iter*unroll+i]);
            fmadd(pY[iter*unroll+i], pY[iter*unroll+i], tmpChunk[i]);
          }
        }

        if( iThread == nThreads - 1 )
        {
          for(long long iChunk = nIter*unroll; iChunk < nChunks; iChunk++)
          {
            fmadd(tmpAlpha, pX[iChunk], pY[iChunk]);
            fmadd(pY[iChunk], pY[iChunk], tmpChunk[iChunk-nIter*unroll]);
          }
        }

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

        constexpr auto unroll = 3;
        const auto nIter = nChunks / unroll;

        const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});

        for(int iCol = 0; iCol < nCols; iCol++)
        {
          const Chunk<T>* pX = &X.chunk(0, iCol);
          Chunk<T>* pY = &Y.chunk(0, iCol);
          const T tmpAlpha = alpha(iCol);
          Chunk<T> tmpChunk[unroll];
          for(int i = 0; i < unroll; i++)
            tmpChunk[i] = Chunk<T>{};

          for(long long iter = firstIter; iter <= lastIter; iter++)
          {
            for(int i = 0; i < unroll; i++)
            {
              fmadd(tmpAlpha, pX[iter*unroll+i], pY[iter*unroll+i]);
              fmadd(pY[iter*unroll+i], pY[iter*unroll+i], tmpChunk[i]);
            }
          }

          if( iThread == nThreads - 1 )
          {
            for(long long iChunk = nIter*unroll; iChunk < nChunks; iChunk++)
            {
              fmadd(tmpAlpha, pX[iChunk], pY[iChunk]);
              fmadd(pY[iChunk], pY[iChunk], tmpChunk[iChunk-nIter*unroll]);
            }
          }

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

  // implement MultiVector axpy+dot
  template<typename T>
  Eigen::ArrayX<T> axpy_dot(const Eigen::ArrayX<T>& alpha, const MultiVector<T>& X, MultiVector<T>& Y, const MultiVector<T>& Z)
  {
    // check dimensions
    if( X.cols() != Y.cols() || X.cols() != Z.cols() || alpha.size() != X.cols() )
      throw std::invalid_argument("MultiVector::axpy_dot: cols dimension mismatch!");

    if( X.rows() != Y.rows() || X.rows() != Z.rows() )
      throw std::invalid_argument("MultiVector::axpy_dot: rows dimension mismatch!");

    const auto nChunks = X.rowChunks();
    const auto nCols = Y.cols();

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"nChunks", "nCols"},{nChunks, nCols}}, // arguments
        {{2*nChunks*nCols*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {2*nChunks*nCols*kernel_info::Load<Chunk<T>>() + nChunks*nCols*kernel_info::Update<Chunk<T>>() + nCols*kernel_info::Store<T>()}} // data transfers
        );

    T tmp[nCols];
    if( nCols == 1 )
    {
      // special case because the OpenMP vector reduction has significant overhead (factor 10)
      T beta = T(0);
#pragma omp parallel reduction(+:beta) if(nChunks > 50)
      {
        auto [iThread,nThreads] = internal::parallel::ompThreadInfo();

        constexpr auto unroll = 3;
        const auto nIter = nChunks / unroll;

        const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});

        const Chunk<T>* pX = &X.chunk(0, 0);
        Chunk<T>* pY = &Y.chunk(0, 0);
        const Chunk<T>* pZ = &Z.chunk(0, 0);
        const T tmpAlpha = alpha(0);
        Chunk<T> tmpChunk[unroll];
        for(int i = 0; i < unroll; i++)
          tmpChunk[i] = Chunk<T>{};

        for(long long iter = firstIter; iter <= lastIter; iter++)
        {
          for(int i = 0; i < unroll; i++)
          {
            fmadd(tmpAlpha, pX[iter*unroll+i], pY[iter*unroll+i]);
            fmadd(pY[iter*unroll+i], pZ[iter*unroll+i], tmpChunk[i]);
          }
        }

        if( iThread == nThreads - 1 )
        {
          for(long long iChunk = nIter*unroll; iChunk < nChunks; iChunk++)
          {
            fmadd(tmpAlpha, pX[iChunk], pY[iChunk]);
            fmadd(pY[iChunk], pZ[iChunk], tmpChunk[iChunk-nIter*unroll]);
          }
        }

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

        constexpr auto unroll = 3;
        const auto nIter = nChunks / unroll;

        const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});

        for(int iCol = 0; iCol < nCols; iCol++)
        {
          const Chunk<T>* pX = &X.chunk(0, iCol);
          Chunk<T>* pY = &Y.chunk(0, iCol);
          const Chunk<T>* pZ = &Z.chunk(0, iCol);
          const T tmpAlpha = alpha(iCol);
          Chunk<T> tmpChunk[unroll];
          for(int i = 0; i < unroll; i++)
            tmpChunk[i] = Chunk<T>{};

          for(long long iter = firstIter; iter <= lastIter; iter++)
          {
            for(int i = 0; i < unroll; i++)
            {
              fmadd(tmpAlpha, pX[iter*unroll+i], pY[iter*unroll+i]);
              fmadd(pY[iter*unroll+i], pZ[iter*unroll+i], tmpChunk[i]);
            }
          }

          if( iThread == nThreads - 1 )
          {
            for(long long iChunk = nIter*unroll; iChunk < nChunks; iChunk++)
            {
              fmadd(tmpAlpha, pX[iChunk], pY[iChunk]);
              fmadd(pY[iChunk], pZ[iChunk], tmpChunk[iChunk-nIter*unroll]);
            }
          }

          for(int i = 0; i < unroll; i++)
            tmp[iCol] += sum(tmpChunk[i]);
        }
      }
    }

    using arr = Eigen::ArrayX<T>;
    arr result = Eigen::Map<arr>(tmp, nCols);
    return result;
  }
}


#endif // PITTS_MULTIVECTOR_AXPBY_IMPL_HPP
