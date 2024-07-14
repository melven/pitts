// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_transform_impl.hpp
* @brief calculate the matrix product of a tall-skinny matrix and a small matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-07-30
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TRANSFORM_IMPL_HPP
#define PITTS_MULTIVECTOR_TRANSFORM_IMPL_HPP

// includes
#include <stdexcept>
#include "pitts_multivector_transform.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement multivector transform
  template<typename T>
  void transform(const MultiVector<T>& X, const ConstTensor2View<T>& M, MultiVector<T>& Y, std::array<long long,2> reshape)
  {
    // check dimensions
    if( X.cols() != M.r1() )
      throw std::invalid_argument("MultiVector::transform: dimension mismatch!");

    if( reshape == std::array<long long,2>{0, 0} )
      reshape = std::array<long long,2>{X.rows(), M.r2()};

    if( reshape[0] * reshape[1] != X.rows()*M.r2() )
      throw std::invalid_argument("MultiVector::transform: invalid reshape dimensions!");

    // check if we can do the fast aligned variant (depends on the reshape dimensions)
    bool fast = X.rows() == reshape[0] ||
                (X.rows() % Chunk<T>::size == 0 && reshape[0] % Chunk<T>::size == 0);

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"Xrows", "Xcols", "Yrows", "Ycols", "fast"},{X.rows(),X.cols(),reshape[0],reshape[1],(long long)fast}}, // arguments
        {{(double(X.rows())*M.r1()*M.r2())*kernel_info::FMA<T>()}, // flops
         {(double(X.rows())*X.cols() + M.r1()*M.r2())*kernel_info::Load<T>() + (double(X.rows())*M.r2())*kernel_info::Store<T>()}} // data transfers
        );

    Y.resize(reshape[0], reshape[1]);
    // special case without reshaping OR where both X and Y have #rows divisible by the chunk size
    if( fast )
    {
#pragma omp parallel for schedule(static)
      for(long long xChunk = 0; xChunk < X.rowChunks(); xChunk++)
      {
        long long yChunk = xChunk % Y.rowChunks();
        long long yj = xChunk / Y.rowChunks();

        long long mj = 0;
        for(; mj+4 < M.r2(); mj+=5)
        {
          Chunk<T> tmp1{}, tmp2{}, tmp3{}, tmp4{}, tmp5{};
          for(long long k = 0; k < M.r1(); k++)
          {
            fmadd(M(k,mj+0), X.chunk(xChunk,k), tmp1);
            fmadd(M(k,mj+1), X.chunk(xChunk,k), tmp2);
            fmadd(M(k,mj+2), X.chunk(xChunk,k), tmp3);
            fmadd(M(k,mj+3), X.chunk(xChunk,k), tmp4);
            fmadd(M(k,mj+4), X.chunk(xChunk,k), tmp5);
          }

          streaming_store(tmp1, Y.chunk(yChunk, yj));
          yChunk += X.rowChunks();
          while( yChunk >= Y.rowChunks() )
          {
            yj++;
            yChunk -= Y.rowChunks();
          }

          streaming_store(tmp2, Y.chunk(yChunk, yj));
          yChunk += X.rowChunks();
          while( yChunk >= Y.rowChunks() )
          {
            yj++;
            yChunk -= Y.rowChunks();
          }

          streaming_store(tmp3, Y.chunk(yChunk, yj));
          yChunk += X.rowChunks();
          while( yChunk >= Y.rowChunks() )
          {
            yj++;
            yChunk -= Y.rowChunks();
          }

          streaming_store(tmp4, Y.chunk(yChunk, yj));
          yChunk += X.rowChunks();
          while( yChunk >= Y.rowChunks() )
          {
            yj++;
            yChunk -= Y.rowChunks();
          }

          streaming_store(tmp5, Y.chunk(yChunk, yj));
          yChunk += X.rowChunks();
          while( yChunk >= Y.rowChunks() )
          {
            yj++;
            yChunk -= Y.rowChunks();
          }
        }

        for(; mj < M.r2(); mj++)
        {
          Chunk<T> tmp{};
          for(long long k = 0; k < M.r1(); k++)
            fmadd(M(k,mj), X.chunk(xChunk,k), tmp);

          streaming_store(tmp, Y.chunk(yChunk, yj));
          yChunk += X.rowChunks();
          while( yChunk >= Y.rowChunks() )
          {
            yj++;
            yChunk -= Y.rowChunks();
          }
        }
      }
      return;
    }

    // generic case where Y is reshaped
#pragma omp parallel for schedule(static)
    for(long long yChunk = 0; yChunk < Y.rowChunks(); yChunk++)
    {
      if( yChunk == Y.rowChunks()-1 )
        continue;
      for(long long yj = 0; yj < Y.cols(); yj++)
      {
        // calculate indices
        const auto index = yChunk*Chunk<T>::size + Y.rows()*yj;
        const auto xi = index % X.rows();
        const auto mj = index / X.rows();
        // non-contiguous case handled later
        if( xi >= (X.rowChunks()-1)*Chunk<T>::size )
          continue;

        // contiguous but possibly unaligned...
        Chunk<T> tmpy{};
        for(long long k = 0; k < M.r1(); k++)
        {
          Chunk<T> tmpx;
          unaligned_load(&X(xi,k), tmpx);
          fmadd(M(k,mj), tmpx, tmpy);
        }
        streaming_store(tmpy, Y.chunk(yChunk,yj));
      }
    }

    // special handling for the last row-chunk of Y (to keep correct zero padding when reshaping)
    if( Y.rowChunks() > 0 )
    {
      const auto lastChunkOffset = (Y.rowChunks()-1)*Chunk<T>::size;
      for(long long yi = lastChunkOffset; yi < Y.rows(); yi++)
      {
        for(long long yj = 0; yj < Y.cols(); yj++)
        {
          // calculate indices
          const auto index = yi + Y.rows()*yj;
          const auto xi = index % X.rows();
          const auto mj = index / X.rows();

          Y(yi,yj) = T(0);
          for(long long k = 0; k < M.r1(); k++)
            Y(yi,yj) += X(xi,k)*M(k,mj);
        }
      }
    }

    // special handling for the last row-chunk of X (as it makes the reshaping more complicated and is omitted above)
    if( X.rowChunks() > 0 && X.rows() != Y.rows() )
    {
      const auto lastChunkOffset = (X.rowChunks()-1)*Chunk<T>::size;
      for(long long xi = lastChunkOffset; xi < X.rows(); xi++)
      {
        for(long long mj = 0; mj < M.r2(); mj++)
        {
          // calculate indices
          const auto index = xi + X.rows()*mj;
          const auto yi = index % Y.rows();
          const auto yj = index / Y.rows();

          Y(yi,yj) = T(0);
          for(long long k = 0; k < M.r1(); k++)
            Y(yi,yj) += X(xi,k)*M(k,mj);
        }
      }
    }

    // special handling for the first row-chunk of X (possibly left out above when reshaping)
    if( X.rowChunks() > 1 && X.rows() != Y.rows() )
    {
      for(long long xi = 0; xi < Chunk<T>::size; xi++)
      {
        for(long long mj = 0; mj < M.r2(); mj++)
        {
          // calculate indices
          const auto index = xi + X.rows()*mj;
          const auto yi = index % Y.rows();
          const auto yj = index / Y.rows();

          Y(yi,yj) = T(0);
          for(long long k = 0; k < M.r1(); k++)
            Y(yi,yj) += X(xi,k)*M(k,mj);
        }
      }
    }

  }


  // implement multivector transform (in-place version)
  template<typename T>
  void transform(MultiVector<T>& X, const ConstTensor2View<T>& M)
  {
    // check dimensions
    if( X.cols() != M.r1() )
      throw std::invalid_argument("MultiVector::transform: dimension mismatch!");

    if( M.r2() > M.r1() )
      throw std::invalid_argument("MultiVector::transform: only supports case M.r1() >= M.r2() !");

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"Xrows", "Xcols", "Xcols_"},{X.rows(),X.cols(),M.r2()}}, // arguments
        {{(double(X.rows())*M.r1()*M.r2())*kernel_info::FMA<T>()}, // flops
         {(double(X.rows())*(M.r1()-M.r2()) + M.r1()*M.r2())*kernel_info::Load<T>() + (double(X.rows())*M.r2())*kernel_info::Update<T>()}} // data transfers
        );


    // dimensions
    const long long nTotalChunks = X.rowChunks();
    const long long ldX = X.colStrideChunks();
    const int m = M.r1();
    const int m_ = M.r2();

    // TODO: blocking for larger m
    // for small m, problem is memory-bound => optimize for simple code, small overhead, n.t.-store where applicable
    // for large m, problem is compute-bound, but only if we handle enough rows at once!

#pragma omp parallel
    {
      std::unique_ptr<Chunk<T>[]> buff(new Chunk<T>[m]);

#pragma omp for schedule(static)
      for(long long iChunk = 0; iChunk < nTotalChunks; iChunk++)
      {
        // offset in input vector
        Chunk<T>* pX = &X.chunk(iChunk, 0);

        int j = 0;
        for(; j+4 <= m_; j+=4)
        {
          Chunk<T> tmp00{};
          Chunk<T> tmp10{};
          Chunk<T> tmp20{};
          Chunk<T> tmp30{};

          for(int k = 0; k < m; k++)
          {
            fmadd(M(k,j+0), pX[k*ldX], tmp00);
            fmadd(M(k,j+1), pX[k*ldX], tmp10);
            fmadd(M(k,j+2), pX[k*ldX], tmp20);
            fmadd(M(k,j+3), pX[k*ldX], tmp30);
          }
          buff[j+0] = tmp00;
          buff[j+1] = tmp10;
          buff[j+2] = tmp20;
          buff[j+3] = tmp30;
        }

        for(; j < m_; j++)
        {
          Chunk<T> tmp00{};

          for(int k = 0; k < m; k++)
            fmadd(M(k,j+0), pX[k*ldX], tmp00);

          buff[j+0] = tmp00;
        }
        // copy buff back to X
        for(int j = 0; j < m; j++)
            pX[j*ldX] = buff[j];
      }
    }

    // shrink to correct size
    X.resize(X.rows(), m_, false, true);
  }

}


#endif // PITTS_MULTIVECTOR_TRANSFORM_IMPL_HPP
