/*! @file pitts_multivector_triangular_solve_impl.hpp
* @brief in-place triangular solve (backward substitution) with a tall-skinny matrix and a small upper triangular matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-03-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TRIANGULAR_SOLVE_IMPL_HPP
#define PITTS_MULTIVECTOR_TRIANGULAR_SOLVE_IMPL_HPP

// includes
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
         {(R.r1()*R.r2())*kernel_info::Load<T>() +
          (double(X.rows())*(R.r2()-streamingCols))*kernel_info::Update<T>() +
          (double(X.rows())*streamingCols)*(kernel_info::Store<T>()+kernel_info::Load<T>())}} // data transfers
        );
    
    // store inverse diagonal
    std::vector<T> invDiag(R.r1());
    for(int i = 0; i < R.r1(); i++)
      invDiag[i] = 1/R(i,i);
    
    constexpr int rowBlockSize = 4;
    const long long nChunks = X.rowChunks();
    const int m = R.r2();
    // TODO: blocking over columns?

#pragma omp parallel
    {
      std::unique_ptr<Chunk<T>[]> buff(new Chunk<T>[m*rowBlockSize]);

#pragma omp for schedule(static)
      for(long long iChunk = 0; iChunk < nChunks; iChunk+=rowBlockSize)
      {
        const long long rowChunks = std::min<long long>(rowBlockSize, nChunks-iChunk);
        for(int j = 0; j < m; j++)
        {
          int piv_j = j;
          if( !colsPermutation.empty() )
            piv_j = colsPermutation[j];
          for(int i = 0; i < rowChunks; i++)
            buff[j*rowBlockSize+i] = X.chunk(iChunk+i, piv_j);
          for(int i = rowChunks; i < rowBlockSize; i++)
            buff[j*rowBlockSize+i] = Chunk<T>{};

          for(int k = 0; k < j; k++)
            for(int i = 0; i < rowBlockSize; i++)
              fnmadd(R(k,j), buff[k*rowBlockSize+i], buff[j*rowBlockSize+i]);
          for(int i = 0; i < rowBlockSize; i++)
            mul(invDiag[j], buff[j*rowBlockSize+i], buff[j*rowBlockSize+i]);
        }
        for(int j = 0; j < m; j++)
          for(int i = 0; i < rowChunks; i++)
          {
            if( streamingStore[j] )
              streaming_store(buff[j*rowBlockSize+i], X.chunk(iChunk+i, j));
            else
              X.chunk(iChunk+i,j) = buff[j*rowBlockSize+i];
          }
      }
    }
    
    // reshape to resulting size
    X.resize(X.rows(), m);
  }

}


#endif // PITTS_MULTIVECTOR_TRIANGULAR_SOLVE_IMPL_HPP
