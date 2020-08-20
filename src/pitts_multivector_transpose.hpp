/*! @file pitts_multivector_transpose.hpp
* @brief reshape and rearrange entries in a tall-skinny matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-08-06
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TRANSPOSE_HPP
#define PITTS_MULTIVECTOR_TRANSPOSE_HPP

// includes
#include "pitts_parallel.hpp"
#include "pitts_multivector.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"
#include <array>
#include <exception>
#include <memory>
#include <tuple>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! reshape and transpose a tall-skinny matrix
  //!
  //! This is equivalent to first adjusting the shape (but keeping the data in the same ordering) and then transposing
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param Y        output mult-vector, resized to dimensions (k, l) or desired shape (see below)
  //! @param reshape  desired new shape (k, l) where l is large and k is small
  //! @param reverse  first transpose, then reshape instead (changes the ordering)
  //!
  template<typename T>
  void transpose(const MultiVector<T>& X, MultiVector<T>& Y, std::array<long long,2> reshape = {0, 0}, bool reverse = false)
  {
    // check dimensions
    if( reshape == std::array<long long,2>{0, 0} )
      reshape = std::array<long long,2>{X.cols(), X.rows()};

    if( reshape[0] * reshape[1] != X.rows()*X.cols() )
      throw std::invalid_argument("MultiVector::transform: invalid reshape dimensions!");


    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"Xrows", "Xcols", "Yrows", "Ycols", "reverse"},{X.rows(),X.cols(),reshape[0],reshape[1],(long long)reverse}}, // arguments
        {{X.rows()*X.cols()*kernel_info::NoOp<T>()}, // flops
         {X.rows()*X.cols()*kernel_info::Load<T>() + reshape[0]*reshape[1]*kernel_info::Store<T>()}} // data transfers
        );

    Y.resize(reshape[0], reshape[1]);
    // difficult to do a nice OpenMP variant as we have two pairs of running indices and easily get idiv-bound when calculating indices in the innermost loop
    if( reverse == false )
    {
#pragma omp parallel
      {
        std::unique_ptr<Chunk<T>[]> buff(new Chunk<T>[reshape[1]]);


        const auto [firstChunk, lastChunk] = internal::parallel::distribute(Y.rowChunks(), internal::parallel::ompThreadInfo());
        const auto indexOffset = firstChunk * Chunk<T>::size * Y.cols();
        long long ix = indexOffset % X.rows();
        long long jx = indexOffset / X.rows();
        for(auto iChunk = firstChunk; iChunk <= lastChunk; iChunk++)
        {
          for(short ii = 0; ii < Chunk<T>::size; ii++)
          {
            if( iChunk == lastChunk && iChunk+1 == Y.rowChunks() && ii + iChunk*Chunk<T>::size >= Y.rows() )
            {
              for(long long jy = 0; jy < Y.cols(); jy++)
                buff[jy][ii] = T(0);
              continue;
            }

            for(long long jy = 0; jy < Y.cols(); jy++)
            {
              buff[jy][ii] = X(ix,jx);

              ix++;
              if( ix == X.rows() )
              {
                jx++;
                ix = 0;
              }
            }
          }
          for(long long jy = 0; jy < Y.cols(); jy++)
            streaming_store(buff[jy], Y.chunk(iChunk,jy));
        }
      }
    }
    else // reverse == true
    {
#pragma omp parallel
      {
        const auto [firstChunk, lastChunk] = internal::parallel::distribute(Y.rowChunks(), internal::parallel::ompThreadInfo());

        for(long long jy = 0; jy < Y.cols(); jy++)
        {
          const auto indexOffset = firstChunk * Chunk<T>::size + jy*Y.rows();
          long long ix = indexOffset / X.cols();
          long long jx = indexOffset % X.cols();

          for(auto iChunk = firstChunk; iChunk <= lastChunk; iChunk++)
          {
            // last chunk is handled below...
            if( iChunk == lastChunk && iChunk == Y.rowChunks()-1 )
              continue;

            Chunk<T> buff;
            for(int ii = 0; ii < Chunk<T>::size; ii++)
            {
              buff[ii] = X(ix,jx);

              jx++;
              if( jx == X.cols() )
              {
                ix++;
                jx = 0;
              }
            }
            streaming_store(buff, Y.chunk(iChunk,jy));
          }

          // handle the last chunk
          if( firstChunk <= lastChunk && lastChunk == Y.rowChunks()-1 )
          {
            const auto iChunk = lastChunk;
            Chunk<T> buff{};
            for(short ii = 0; ii < Chunk<T>::size && ii + iChunk*Chunk<T>::size < Y.rows(); ii++)
            {
              buff[ii] = X(ix,jx);

              jx++;
              if( jx == X.cols() )
              {
                ix++;
                jx = 0;
              }
            }

            streaming_store(buff, Y.chunk(iChunk,jy));
          }
        }
      }
    }
  }
}

#endif // PITTS_MULTIVECTOR_TRANSPOSE_HPP
