// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_triangular_solve_impl.hpp
* @brief in-place triangular solve (backward substitution) with a tall-skinny matrix and a small upper triangular matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-03-26
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TRIANGULAR_SOLVE_IMPL_HPP
#define PITTS_MULTIVECTOR_TRIANGULAR_SOLVE_IMPL_HPP

// includes
#include <omp.h>
#include <stdexcept>
#include <memory>
#include "pitts_multivector_triangular_solve.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement multivector triangular_solve
  template<typename T>
  void triangularSolve(MultiVector<T>& X, const Tensor2<T>& R, const std::vector<int>& colsPermutation)
  {
    // special case: zero rank => dimensions are zero
    if( R.r1() == 0 && R.r2() == 0 && colsPermutation.empty() )
    {
      X.resize(X.rows(), 0);
      return;
    }

    if( colsPermutation.empty() )
    {
      // no column permutation: matrix dimensions must match
      if( R.r1() != R.r2() || R.r1() != X.cols() )
        throw std::invalid_argument("MultiVector::triangularSolve: dimension mismatch!");
    }
    else
    {
      // with column permutation: only use subset of columns of X
      if( R.r1() != R.r2() || R.r1() != colsPermutation.size() || R.r1() > X.cols() )
        throw std::invalid_argument("MultiVector::triangularSolve: dimension mismatch!");
      
      // check that the indices in colsPermutation are valid
      for(const int idx: colsPermutation)
      {
        if( idx >= X.cols() )
          throw std::invalid_argument("MultiVector::triangularSolve: invalid column permutation index!");
      }
    }

    // perform streaming store for column i if it is not read
    const std::vector<bool> streamingStore = [&]()
    {
      if( colsPermutation.empty() )
        return std::vector<bool>(X.cols(), false);
      
      std::vector<bool> tmp(X.cols(), true);
      for(const int idx: colsPermutation)
        tmp[idx] = false;
      return tmp;
    }();
    // amount of data updated (if cols > r1)
    const int streamingCols = [&]()
    {
      int tmp = 0;
      for(int i = 0; i < R.r2(); i++)
        tmp += (int)streamingStore[i];
      return tmp;
    }();

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"Xrows", "Xcols", "r", "streamingCols"},{X.rows(),X.cols(),R.r1(),streamingCols}}, // arguments
        {{(0.5*X.rows()*R.r1()*R.r2())*kernel_info::FMA<T>()}, // flops
         {(0.5*R.r1()*R.r2())*kernel_info::Load<T>() +
          (double(X.rows())*(R.r2()-streamingCols))*kernel_info::Update<T>() +
          (double(X.rows())*streamingCols)*(kernel_info::Store<T>()+kernel_info::Load<T>())}} // data transfers
        );
    
    // store inverse diagonal
    std::vector<T> invDiag(R.r1());
    for(int i = 0; i < R.r1(); i++)
      invDiag[i] = 1/R(i,i);
    
    // dimensions
    const long long nTotalChunks = X.rowChunks();
    const int m = R.r2();
    // for small m, problem is memory-bound => optimize for simple code, small overhead, n.t.-store where applicable
    // for large m, problem is compute-bound, but only if we handle enough rows at once!
    constexpr int nChunks = 10;
    //int nMaxThreads = omp_get_max_threads();
    //nChunks = std::min<long long>(nChunks, nTotalChunks / (2*nMaxThreads));
    //nChunks = std::max(1, nChunks);
    
    constexpr int rowBlockSize = 2;
    constexpr int colBlockSize = 60;

#pragma omp parallel
    {
      std::unique_ptr<Chunk<T>[]> buff(new Chunk<T>[m*nChunks]);

#pragma omp for schedule(static)
      for(long long iChunk = 0; iChunk < nTotalChunks; iChunk+=nChunks)
      {
        const int nRemainingChunks = std::min<long long>(nChunks, nTotalChunks-iChunk);

        // copy data to buff
        for(int j = 0; j < m; j++)
        {
          int piv_j = j;
          if( !colsPermutation.empty() )
            piv_j = colsPermutation[j];
          for(int i = 0; i < nRemainingChunks; i++)
            buff[j*nChunks+i] = X.chunk(iChunk+i, piv_j);
          for(int i = nRemainingChunks; i < nChunks; i++)
            buff[j*nChunks+i] = Chunk<T>{};
        }

        // perform calculation with buff
        for(int jb = 0; jb < m; jb+=colBlockSize)
        {
          // blocks above the diagonal
          for(int kb = 0; kb < jb; kb+=colBlockSize)
            for(int ib = 0; ib < nChunks; ib+=rowBlockSize)
            {
              int j = jb;
              for(; j+3 < std::min(m, jb+colBlockSize); j+=4)
              {
                Chunk<T> tmp00 = buff[(j+0)*nChunks+ib+0];
                Chunk<T> tmp10 = buff[(j+0)*nChunks+ib+1];
                Chunk<T> tmp01 = buff[(j+1)*nChunks+ib+0];
                Chunk<T> tmp11 = buff[(j+1)*nChunks+ib+1];
                Chunk<T> tmp02 = buff[(j+2)*nChunks+ib+0];
                Chunk<T> tmp12 = buff[(j+2)*nChunks+ib+1];
                Chunk<T> tmp03 = buff[(j+3)*nChunks+ib+0];
                Chunk<T> tmp13 = buff[(j+3)*nChunks+ib+1];
                for(int k = kb; k < kb+colBlockSize; k++)
                {
                  fnmadd(R(k,j+0), buff[k*nChunks+ib+0], tmp00);
                  fnmadd(R(k,j+0), buff[k*nChunks+ib+1], tmp10);
                  fnmadd(R(k,j+1), buff[k*nChunks+ib+0], tmp01);
                  fnmadd(R(k,j+1), buff[k*nChunks+ib+1], tmp11);
                  fnmadd(R(k,j+2), buff[k*nChunks+ib+0], tmp02);
                  fnmadd(R(k,j+2), buff[k*nChunks+ib+1], tmp12);
                  fnmadd(R(k,j+3), buff[k*nChunks+ib+0], tmp03);
                  fnmadd(R(k,j+3), buff[k*nChunks+ib+1], tmp13);
                }
                buff[(j+0)*nChunks+ib+0] = tmp00;
                buff[(j+0)*nChunks+ib+1] = tmp10;
                buff[(j+1)*nChunks+ib+0] = tmp01;
                buff[(j+1)*nChunks+ib+1] = tmp11;
                buff[(j+2)*nChunks+ib+0] = tmp02;
                buff[(j+2)*nChunks+ib+1] = tmp12;
                buff[(j+3)*nChunks+ib+0] = tmp03;
                buff[(j+3)*nChunks+ib+1] = tmp13;
              }
              for(; j < std::min(m, jb+colBlockSize); j++)
              {
                Chunk<T> tmp0 = buff[j*nChunks+ib+0];
                Chunk<T> tmp1 = buff[j*nChunks+ib+1];
                for(int k = kb; k < kb+colBlockSize; k++)
                {
                  fnmadd(R(k,j), buff[k*nChunks+ib+0], tmp0);
                  fnmadd(R(k,j), buff[k*nChunks+ib+1], tmp1);
                }
                buff[j*nChunks+ib+0] = tmp0;
                buff[j*nChunks+ib+1] = tmp1;
              }
            }
          // diagonal block
          {
            const int kb = jb;
            for(int j = jb; j < std::min(m, jb+colBlockSize); j++)
            {
              for(int k = kb; k < j; k++)
                for(int i = 0; i < nChunks; i++)
                  fnmadd(R(k,j), buff[k*nChunks+i], buff[j*nChunks+i]);
              for(int i = 0; i < nChunks; i++)
                mul(invDiag[j], buff[(j+0)*nChunks+i], buff[(j+0)*nChunks+i]);
            }
          }
        }

        // store result
        for(int j = 0; j < m; j++)
          for(int i = 0; i < nRemainingChunks; i++)
          {
            if( streamingStore[j] )
              streaming_store(buff[j*nChunks+i], X.chunk(iChunk+i, j));
            else
              X.chunk(iChunk+i,j) = buff[j*nChunks+i];
          }
      }
    }
    
    // reshape to resulting size
    X.resize(X.rows(), m);
  }

}


#endif // PITTS_MULTIVECTOR_TRIANGULAR_SOLVE_IMPL_HPP
