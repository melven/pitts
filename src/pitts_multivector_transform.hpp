/*! @file pitts_multivector_transform.hpp
* @brief calculate the matrix product of a tall-skinny matrix and a small matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-07-30
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TRANSFORM_HPP
#define PITTS_MULTIVECTOR_TRANSFORM_HPP

// includes
#include <array>
#include <exception>
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the matrix-matrix product of a tall-skinny matrix (multivector) with a small matrix (Y <- X*M)
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param M        small transformation matrix, dimensions (m, k)
  //! @param Y        resulting mult-vector, resized to dimensions (n, k) or desired shape (see below)
  //! @param reshape  desired shape of the resulting multi-vector, total size must be n*k
  //!
  template<typename T>
  void transform(const MultiVector<T>& X, const Tensor2<T>& M, MultiVector<T>& Y, std::array<long long,2> reshape = {0, 0})
  {
    // check dimensions
    if( X.cols() != M.r1() )
      throw std::invalid_argument("MultiVector::transform: dimension mismatch!");

    if( reshape == std::array<long long,2>{0, 0} )
      reshape = std::array<long long,2>{X.rows(), M.r2()};

    if( reshape[0] * reshape[1] != X.rows()*M.r2() )
      throw std::invalid_argument("MultiVector::transform: invalid reshape dimensions!");

    // check if we can do the fast aligned variant (depends on the reshape dimensions)
    bool fast = X.rows() == Y.rows() ||
                (X.rows() % Chunk<T>::size == 0 && Y.rows() % Chunk<T>::size == 0);

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
        for(; mj+3 < M.r2(); mj+=4)
        {
          Chunk<T> tmp1{}, tmp2{}, tmp3{}, tmp4{};
          for(long long k = 0; k < M.r1(); k++)
          {
            fmadd(M(k,mj+0), X.chunk(xChunk,k), tmp1);
            fmadd(M(k,mj+1), X.chunk(xChunk,k), tmp2);
            fmadd(M(k,mj+2), X.chunk(xChunk,k), tmp3);
            fmadd(M(k,mj+3), X.chunk(xChunk,k), tmp4);
          }

          streaming_store(tmp1, Y.chunk(yChunk, yj));
          yChunk += X.rowChunks();
          while( yChunk > Y.rowChunks() )
          {
            yj++;
            yChunk -= Y.rowChunks();
          }

          streaming_store(tmp2, Y.chunk(yChunk, yj));
          yChunk += X.rowChunks();
          while( yChunk > Y.rowChunks() )
          {
            yj++;
            yChunk -= Y.rowChunks();
          }

          streaming_store(tmp3, Y.chunk(yChunk, yj));
          yChunk += X.rowChunks();
          while( yChunk > Y.rowChunks() )
          {
            yj++;
            yChunk -= Y.rowChunks();
          }

          streaming_store(tmp4, Y.chunk(yChunk, yj));
          yChunk += X.rowChunks();
          while( yChunk > Y.rowChunks() )
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
          while( yChunk > Y.rowChunks() )
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

}


#endif // PITTS_MULTIVECTOR_TRANSFORM_HPP
