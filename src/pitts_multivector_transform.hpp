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
  void transform(const MultiVector<T>& X, const Tensor2<T>& M, MultiVector<T>& Y, std::array<int,2> reshape = {0, 0})
  {
    // check dimensions
    if( X.cols() != M.r1() )
      throw std::invalid_argument("MultiVector::transform: dimension mismatch!");

    if( reshape == std::array<int,2>{0, 0} )
      reshape = std::array<int,2>{X.rows(), M.r2()};

    if( reshape[0] * reshape[1] != X.rows()*M.r2() )
      throw std::invalid_argument("MultiVector::transform: invalid reshape dimensions!");


    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"Xrows", "Xcols", "Yrows", "Ycols"},{X.rows(),X.cols(),reshape[0],reshape[1]}}, // arguments
        {{(double(X.rows())*M.r1()*M.r2())*kernel_info::FMA<T>()}, // flops
         {(double(X.rows())*X.cols() + M.r1()*M.r2())*kernel_info::Load<T>() + (double(X.rows())*M.r2())*kernel_info::Store<T>()}} // data transfers
        );

    Y.resize(reshape[0], reshape[1]);
#pragma omp parallel for schedule(static)
    for(int yChunk = 0; yChunk < Y.rowChunks()-1; yChunk++)
    {
      for(int yj = 0; yj < Y.cols(); yj++)
      {
        // calculate indices
        const auto index = yChunk*Chunk<T>::size + ((long long)(Y.rows()))*yj;
        const auto xi = int(index % X.rows());
        const auto mj = int(index / X.rows());
        // non-contiguous case handled later
        if( xi >= (X.rowChunks()-1)*Chunk<T>::size )
          continue;

        // contiguous but possibly unaligned...
        Chunk<T> tmpy{};
        for(int k = 0; k < M.r1(); k++)
        {
          Chunk<T> tmpx;
          for(int ii = 0; ii < Chunk<T>::size; ii++)
            tmpx[ii] = *(&X(xi,k)+ii);
          fmadd(M(k,mj), tmpx, tmpy);
        }
        // TODO stream if large...
        Y.chunk(yChunk,yj) = tmpy;
      }
    }

    // special handling for the last row-chunk of Y (to keep correct zero padding when reshaping)
    if( Y.rowChunks() > 0 )
    {
      const auto lastChunkOffset = (Y.rowChunks()-1)*Chunk<T>::size;
      for(int yi = lastChunkOffset; yi < Y.rows(); yi++)
      {
        for(int yj = 0; yj < Y.cols(); yj++)
        {
          // calculate indices
          const auto index = yi + ((long long)(Y.rows()))*yj;
          const auto xi = int(index % X.rows());
          const auto mj = int(index / X.rows());

          Y(yi,yj) = T(0);
          for(int k = 0; k < M.r1(); k++)
            Y(yi,yj) += X(xi,k)*M(k,mj);
        }
      }
    }

    // special handling for the last row-chunk of X (as it makes the reshaping more complicated and is omitted above)
    if( X.rowChunks() > 0 && X.rows() != Y.rows() )
    {
      const auto lastChunkOffset = (X.rowChunks()-1)*Chunk<T>::size;
      for(int xi = lastChunkOffset; xi < X.rows(); xi++)
      {
        for(int mj = 0; mj < M.r2(); mj++)
        {
          // calculate indices
          const auto index = xi + ((long long)(X.rows()))*mj;
          const auto yi = int(index % Y.rows());
          const auto yj = int(index / Y.rows());

          Y(yi,yj) = T(0);
          for(int k = 0; k < M.r1(); k++)
            Y(yi,yj) += X(xi,k)*M(k,mj);
        }
      }
    }

    // special handling for the first row-chunk of X (possibly left out above when reshaping)
    if( X.rowChunks() > 1 && X.rows() != Y.rows() )
    {
      for(int xi = 0; xi < Chunk<T>::size; xi++)
      {
        for(int mj = 0; mj < M.r2(); mj++)
        {
          // calculate indices
          const auto index = xi + ((long long)(X.rows()))*mj;
          const auto yi = int(index % Y.rows());
          const auto yj = int(index / Y.rows());

          Y(yi,yj) = T(0);
          for(int k = 0; k < M.r1(); k++)
            Y(yi,yj) += X(xi,k)*M(k,mj);
        }
      }
    }

  }

}


#endif // PITTS_MULTIVECTOR_TRANSFORM_HPP
