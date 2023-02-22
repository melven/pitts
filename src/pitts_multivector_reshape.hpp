/*! @file pitts_multivector_reshape.hpp
* @brief adjust the shape of a tall-skinny matrix keeping the data
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-06-29
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_RESHAPE_HPP
#define PITTS_MULTIVECTOR_RESHAPE_HPP

// includes
#include <stdexcept>
#include "pitts_multivector.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! reshape a tall-skinny matrix (multivector)
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param rows     new number of rows, rows*cols must be equal to n*m
  //! @param cols     new number of columns, rows*cols must be equal to n*m
  //! @param Y        resulting multi-vector, resized to (rows, cols)
  //!
  template<typename T>
  void reshape(const MultiVector<T>& X, long long rows, long long cols, MultiVector<T>& Y)
  {
    // check dimensions
    if( rows*cols != X.rows()*X.cols() )
      throw std::invalid_argument("MultiVector::reshape: invalid new dimensions!");
    
    if( X.rows() == rows && X.cols() == cols )
    {
      // perform a copy instead...
      copy(X, Y);
      return;
    }

    // check if we can do the fast aligned variant (depends on the reshape dimensions)
    bool fast = (X.rows() % Chunk<T>::size == 0 && rows % Chunk<T>::size == 0);

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"Xrows", "Xcols", "Yrows", "Ycols", "fast"},{X.rows(),X.cols(),rows,cols,(long long)fast}}, // arguments
        {{X.rows()*X.cols()*kernel_info::NoOp<T>()}, // flops
         {double(X.rows())*X.cols()*kernel_info::Load<T>() + double(rows)*cols*kernel_info::Store<T>()}} // data transfers
        );

    Y.resize(rows, cols);
    // special case without reshaping OR where both X and Y have #rows divisible by the chunk size
    if( fast )
    {
#pragma omp parallel for schedule(static)
      for(long long yChunk = 0; yChunk < Y.rowChunks(); yChunk++)
      {
        for(long long yj = 0; yj < cols; yj++)
        {
          // yChunk + yj*Y.rowChunks() == xChunk + xj*X.rowChunks()
          const auto flatIdx = yChunk + yj*Y.rowChunks();
          const auto xChunk = flatIdx % X.rowChunks();
          const auto xj = flatIdx / X.rowChunks();

          streaming_store(X.chunk(xChunk,xj), Y.chunk(yChunk,yj));
        }
      }
      return;
    }

    // generic case where new old/new number of rows are not multiples of the chunk size
#pragma omp parallel for schedule(static)
    for(long long yChunk = 0; yChunk < Y.rowChunks(); yChunk++)
    {
      if( yChunk == Y.rowChunks()-1 )
        continue;
      for(long long yj = 0; yj < cols; yj++)
      {
        // yChunk + yj*Y.rowChunks() == xChunk + xj*X.rowChunks()
        const auto flatIdx = yChunk*Chunk<T>::size + yj*Y.rows();
        const auto xi = flatIdx % X.rows();
        const auto xj = flatIdx / X.rows();
        // non-contiguous case handled later
        if( xi >= (X.rowChunks()-1)*Chunk<T>::size )
          continue;

        // contiguous but possibly unaligned...
        Chunk<T> tmp;
        unaligned_load(&X(xi,xj), tmp);

        streaming_store(tmp, Y.chunk(yChunk,yj));
      }
    }

    // special handling for the last row-chunk of Y (to keep correct zero padding while reshaping)
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
          const auto xj = index / X.rows();

          Y(yi,yj) = X(xi,xj);
        }
      }
    }

    // special handling for the last row-chunk of X (as it makes the reshaping more complicated and is omitted above)
    if( X.rowChunks() > 0 )
    {
      const auto lastChunkOffset = (X.rowChunks()-1)*Chunk<T>::size;
      for(long long xi = lastChunkOffset; xi < X.rows(); xi++)
      {
        for(long long xj = 0; xj < X.cols(); xj++)
        {
          // calculate indices
          const auto index = xi + X.rows()*xj;
          const auto yi = index % Y.rows();
          const auto yj = index / Y.rows();

          Y(yi,yj) = X(xi,xj);
        }
      }
    }

    // special handling for the first row-chunk of X (possibly left out above)
    if( X.rowChunks() > 1 )
    {
      for(long long xi = 0; xi < Chunk<T>::size; xi++)
      {
        for(long long xj = 0; xj < X.cols(); xj++)
        {
          // calculate indices
          const auto index = xi + X.rows()*xj;
          const auto yi = index % Y.rows();
          const auto yj = index / Y.rows();

          Y(yi,yj) = X(xi,xj);
        }
      }
    }
  }

}


#endif // PITTS_MULTIVECTOR_RESHAPE_HPP
