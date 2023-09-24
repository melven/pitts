/*! @file pitts_multivector_tsqr_impl.hpp
* @brief calculate the QR-decomposition of a multi-vector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-07-13
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
* Based on
* Demmel et.al.: "Communication-optimal Parallel and Sequential QR and LU Factorizations", SISC 2012, doi 10.1137/080731992
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TSQR_IMPL_HPP
#define PITTS_MULTIVECTOR_TSQR_IMPL_HPP

// includes
#include <cassert>
#include <memory>
#include <vector>
#include <cstdint>
#include <cmath>
#include <bit>
#include <latch>
#include <array>
#include "pitts_multivector_tsqr.hpp"
#include "pitts_parallel.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! wrapper type for std::latch that provides default-initialization
    class Latch : public std::latch
    {
        public:
            Latch(std::ptrdiff_t expected = 0) : std::latch(expected) {}
    };

    // helper type with 3 std::latch'es
    using LatchArray3 = std::array<Latch, 3>;

    //! helper function to reinitialize an std::latch (ugly)
    inline void resetLatch(Latch& l, std::ptrdiff_t expected)
    {
        l.~Latch();
        new(&l) Latch(expected);
    }

    //! helper namespace for TSQR routines
    namespace HouseholderQR
    {
      //! Apply a Householder reflection of the form (I - v v^T) with ||v|| = sqrt(2)
      //!
      //! Usually, a Householder reflection has the form (I - 2 v v^T), the factor 2 is included in v to improve the performance.
      //!
      //! This function assumes a very specific memory layout of the data (with a chunk size cs):
      //!
      //! @verbatim
      //!  nChunks*cs rows    * * * * * *
      //!  in pdata:          * * * * * *
      //!
      //!  triangular form    x x x x x x
      //!  of pdataResult:    0 x x x x x
      //!                     0 0 x x x x
      //!                     0 0 0 x x x
      //!                     0 0 0 0 x x
      //!                     0 0 0 0 0 x
      //! @endverbatim
      //!
      //! When calculating a QR decomposition of this matrix, the reflection vectors v have only nChunks*cs non-zero entries (starting with chunk firstRow).
      //!
      //! @tparam T           underlying data type
      //!
      //! @param nChunks      number of non-zero rows in v (divided by the Chunk size)
      //! @param firstRow     index of the chunk that contains the current pivot element (0 <= firstRow)
      //! @param col          current column index
      //! @param v            Householder vector with norm sqrt(2), only considering the chunks [firstRow:firstRow+nChunks]
      //! @param pdata        column-major input array with dimension lda*#columns; can be identical to pdataResult for in-place calculation
      //! @param lda          offset of columns in pdata
      //! @param pdataResult  dense, column-major output array with dimension ldaResult*#columns, the upper firstRow-1 chunks of rows are not touched
      //! @param ldaResult    offset of columns in pdataResult
      //!
      template<typename T>
      [[gnu::always_inline]]
      inline void applyReflection(int nChunks, int firstRow, int col, const Chunk<T>* v, const Chunk<T>* pdata, long long lda, Chunk<T>* pdataResult, int ldaResult)
      {
//std::cout << "apply " << col << "\n";
        if( pdata == pdataResult || firstRow >= nChunks )
        {
          // fast case: in-place operation
          Chunk<T> vTx{};
          for(int i = firstRow; i <= nChunks+firstRow; i++)
            fmadd(v[i], pdataResult[i+ldaResult*col], vTx);
          bcast_sum(vTx);
          for(int i = firstRow; i <= nChunks+firstRow; i++)
            fnmadd(vTx, v[i], pdataResult[i+ldaResult*col], pdataResult[i+ldaResult*col]);
        }
        else
        {
          // generic case: out-of-place operation with input data from both pdata and pdataResult
          Chunk<T> vTx{};
          for(int i = firstRow; i < nChunks; i++)
            fmadd(v[i], pdata[i+lda*col], vTx);
          for(int i = nChunks; i <= nChunks+firstRow; i++)
            fmadd(v[i], pdataResult[i+ldaResult*col], vTx);
          bcast_sum(vTx);
          for(int i = firstRow; i < nChunks; i++)
            fnmadd(vTx, v[i], pdata[i+lda*col], pdataResult[i+ldaResult*col]);
          for(int i = nChunks; i <= nChunks+firstRow; i++)
            fnmadd(vTx, v[i], pdataResult[i+ldaResult*col], pdataResult[i+ldaResult*col]);
        }
      }


      //! Apply two consecutive Householder reflection of the form (I - v v^T) (I - w w^T) with ||v|| = ||w|| = sqrt(2) to possibly multiple columns
      //!
      //! Usually, a Householder reflection has the form (I - 2 v v^T), the factor 2 is included in v and w to improve the performance.
      //!
      //! This function has the same effect as calling applyReflection twice, first with vector w and then with vector v.
      //! Using this function avoids to transfer required colums to/from the cache twice.
      //!
      //! Exploits (I - v v^T) (I -w w^T) = I - v (v^T - v^T w w^T) - w w^T where v^T w can be calculated in advance.
      //!
      //! See applyReflection for details on the assumed memory layout.
      //!
      //! @tparam T           underlying data type
      //! @tparam NC          column unroll factor
      //!
      //! @param nChunks      number of non-zero rows in v (divided by the Chunk size)
      //! @param firstRow     index of the chunk that contains the current pivot element (0 <= firstRow)
      //! @param col          index of the first of NC consecutive columns.
      //! @param w            first Householder vector with norm sqrt(2), only considers the chunks [firstRow:firstRow+nChunks]
      //! @param v            second Householder vector with norm sqrt(2), only considers the chunks [firstRow:firstRow+nChunks]
      //! @param vTw          scalar product of v and w; required as we apply both transformations at once
      //! @param pdata        column-major input array with dimension lda*#columns; can be identical to pdataResult for in-place calculation
      //! @param lda          offset of columns in pdata
      //! @param pdataResult  dense, column-major output array with dimension ldaResult*#columns, the upper firstRow-1 chunks of rows are not touched
      //! @param ldaResult    offset of columns in pdataResult
      //!
      template<typename T, int NC = 1>
      [[gnu::always_inline]]
      inline void applyReflection2(int nChunks, int firstRow, int col, const Chunk<T>* w, const Chunk<T>* v, const Chunk<T> &vTw, const Chunk<T>* pdata, long long lda, Chunk<T>* pdataResult, int ldaResult)
      {
//for(int j = 0; j < NC; j++)
//  std::cout << "apply " << col+j << "\n";

        Chunk<T> wTx[NC];
        Chunk<T> vTx[NC];
        for(int j = 0; j < NC; j++)
          wTx[j] = vTx[j] = Chunk<T>{};
        if( pdata == pdataResult || firstRow >= nChunks )
        {
          // fast case: in-place operation
          for(int i = firstRow; i <= nChunks+firstRow; i++)
          {
            for(int j = 0; j < NC; j++)
            {
              fmadd(w[i], pdataResult[i+ldaResult*(col+j)], wTx[j]);
              fmadd(v[i], pdataResult[i+ldaResult*(col+j)], vTx[j]);
            }
          }
          for(int j = 0; j < NC; j++)
          {
            bcast_sum(wTx[j]);
            bcast_sum(vTx[j]);
          }
          for(int j = 0; j < NC; j++)
          {
            fnmadd(vTw, wTx[j], vTx[j]);
          }
          for(int i = firstRow; i <= nChunks+firstRow; i++)
          {
            Chunk<T> tmp[NC];
            for(int j = 0; j < NC; j++)
              tmp[j] = pdataResult[i+ldaResult*(col+j)];

            for(int j = 0; j < NC; j++)
            {
              fnmadd(wTx[j], w[i], tmp[j]);
              fnmadd(vTx[j], v[i], tmp[j]);
            }

            for(int j = 0; j < NC; j++)
              pdataResult[i+ldaResult*(col+j)] = tmp[j];
          }
        }
        else
        {
          // generic case: out-of-place operation with input data from both pdata and pdataResult
          for(int i = firstRow; i < nChunks; i++)
          {
            for(int j = 0; j < NC; j++)
            {
              fmadd(w[i], pdata[i+lda*(col+j)], wTx[j]);
              fmadd(v[i], pdata[i+lda*(col+j)], vTx[j]);
            }
          }
          for(int i = nChunks; i <= nChunks+firstRow; i++)
          {
            for(int j = 0; j < NC; j++)
            {
              fmadd(w[i], pdataResult[i+ldaResult*(col+j)], wTx[j]);
              fmadd(v[i], pdataResult[i+ldaResult*(col+j)], vTx[j]);
            }
          }
          for(int j = 0; j < NC; j++)
          {
            bcast_sum(wTx[j]);
            bcast_sum(vTx[j]);
          }
          for(int j = 0; j < NC; j++)
          {
            fnmadd(vTw, wTx[j], vTx[j]);
          }
          for(int i = firstRow; i < nChunks; i++)
          {
            Chunk<T> tmp[NC];
            for(int j = 0; j < NC; j++)
              tmp[j] = pdata[i+lda*(col+j)];

            for(int j = 0; j < NC; j++)
            {
              fnmadd(wTx[j], w[i], tmp[j]);
              fnmadd(vTx[j], v[i], tmp[j]);
            }

            for(int j = 0; j < NC; j++)
              pdataResult[i+ldaResult*(col+j)] = tmp[j];
          }
          for(int i = nChunks; i <= nChunks+firstRow; i++)
          {
            Chunk<T> tmp[NC];
            for(int j = 0; j < NC; j++)
              tmp[j] = pdataResult[i+ldaResult*(col+j)];

            for(int j = 0; j < NC; j++)
            {
              fnmadd(wTx[j], w[i], tmp[j]);
              fnmadd(vTx[j], v[i], tmp[j]);
            }

            for(int j = 0; j < NC; j++)
              pdataResult[i+ldaResult*(col+j)] = tmp[j];
          }
        }
      }

      // forward declaration
      template<typename T>
      [[gnu::always_inline]]
      inline void transformBlock_calc(int nChunks, int m, const Chunk<T>* pdataIn, long long ldaIn, Chunk<T>* pdataResult, int ldaResult, int resultOffset, int col);

      // forward declaration
      template<typename T>
      [[gnu::always_inline]]
      inline void transformBlock_apply(int nChunks, int m, const Chunk<T>* pdataIn, long long ldaIn, Chunk<T>* pdataResult, int ldaResult, int resultOffset, int beginCol, int endCol, int applyBeginCol, int applyEndCol);



      //! Calculate the upper triangular part R from a QR-decomposition of a small rectangular block where the bottom left triangle is already zero
      //!
      //! Can work in-place or out-of-place.
      //!
      //! This function assumes a very specific memory layout of the data (with a chunk size cs):
      //!
      //! @verbatim
      //!  nChunks*cs rows    * * * * * *                             x x x x x x
      //!  in pdata:          * * * * * *                             0 x x x x x
      //!                                    -- transformed to -->
      //!  triangular form    x x x x x x                             0 0 x x x x
      //!  of pdataResult:    0 x x x x x                             0 0 0 x x x
      //!                     0 0 x x x x                             0 0 0 0 x x
      //!                     0 0 0 x x x                             0 0 0 0 0 x
      //!                     0 0 0 0 x x                             0 0 0 0 0 0
      //!                     0 0 0 0 0 x                             0 0 0 0 0 0
      //! @endverbatim
      //!
      //! @warning For out-of-place (pdataIn != pdataResult) calculation, the triangular result is copied back to the bottom of the buffer.
      //!          For in-place (pdataIn == pdataResult) calculation, the triangular result is at the top of the buffer.
      //!
      //! @warning This assumes additional available memory around pdataResult for one additional chunk row
      //           in front of (out-of-place: pdataResult != pdataIn), respectively behind (in-place: pdataResult == pdataIn) the current block.
      //!
      //! @tparam T           underlying data type
      //!
      //! @param nChunks      number of new rows in pdataIn divided by the Chunk size
      //! @param m            number of columns
      //! @param pdataIn      column-major input array with dimension ldaIn*m; can be identical to pdataResult for in-place calculation
      //! @param ldaIn        offset of columns in pdata
      //! @param pdataResult  dense, column-major output array with dimension ldaResult*m; contains the upper triangular R on exit; the lower triangular part is set to zero.
      //!                     On input, the bottom part must be upper triangular, the first nChunk chunks of rows are ignored.
      //! @param ldaResult    offset of columns in pdataResult
      //! @param resultOffset row chunk offset of the triangular part on input in pdataResult, expected to be >= nChunks+2 for out-of-place calculation and nChunks for in-place calculation
      //! @param colBlockSize tuning parameter for better cache access if m is large, should be a multiple of 3 (because of some internal 3-way unrolling)
      //!
      /*
      template<typename T>
      void transformBlock(int nChunks, int m, const Chunk<T>* pdataIn, long long ldaIn, Chunk<T>* pdataResult, int ldaResult, int resultOffset, int colBlockSize = 15)
      {
        const int mChunks = (m-1) / Chunk<T>::size + 1;
        // we need enough buffer space because we store some additional vectors in it...
        if( pdataIn == pdataResult )
        {
          // in-place
          assert(resultOffset == nChunks);
          assert(ldaResult >= mChunks + nChunks + 2);
        }
        else
        {
          // out-of-place
          assert(resultOffset >= nChunks + 2);
          assert(ldaResult >= mChunks + resultOffset);
          pdataResult = pdataResult + resultOffset - nChunks;
        }

        // this is an approach for hierarchical blocking of
        // for(int i = 0; i < m; i++)
        //   calc i
        //   for(int j = i+1; j < m; j++)
        //     apply i to j

        const int bs = colBlockSize;

        const auto tree_apply = [&](const auto& tree_apply, int beginCol, int endCol, int applyBeginCol, int applyEndCol) -> void
        {
          int nCol = endCol - beginCol;
          // could also split by nApplyCol but doesn't seem to be help
          //int nApplyCol = applyEndCol - applyBeginCol;

          if( nCol < 2*bs )
          {
            transformBlock_apply(nChunks, m, pdataIn, ldaIn, pdataResult, ldaResult, resultOffset, beginCol, endCol, applyBeginCol, applyEndCol);
          }
          else
          {
            int middle = beginCol + (nCol/2/bs)*bs;
            tree_apply(tree_apply, beginCol, middle, applyBeginCol, applyEndCol);
            tree_apply(tree_apply, middle, endCol, applyBeginCol, applyEndCol);
          }
        };

        const auto tree_calc = [&](const auto& tree_calc, int beginCol, int endCol) -> void
        {
          int nCol = endCol - beginCol;
          if( nCol < 2*bs )
          {
            for(int col = beginCol; col < endCol; col++)
            {
              transformBlock_calc(nChunks, m, pdataIn, ldaIn, pdataResult, ldaResult, resultOffset, col);
              transformBlock_apply(nChunks, m, pdataIn, ldaIn, pdataResult, ldaResult, resultOffset, col, col+1, col+1, endCol);
            }
          }
          else
          {
            int middle = beginCol + (nCol/2/bs)*bs;
            tree_calc(tree_calc, beginCol, middle);
            tree_apply(tree_apply, beginCol, middle, middle, endCol);
            tree_calc(tree_calc, middle, endCol);
          }
        };

        tree_calc(tree_calc,0,m);
      }
      */

      //! Same as transformBlock, but implementing a parallelization of tree_apply over lastThread-firstThread OMP threads with contiguous thread id's.
      //! 
      //! All participating threads [firstThread,lastThread) need to call this function (with the same arguments)!
      //!
      //! @param firstThread  OMP thread id of first thread
      //! @param lastThread   OMP thread of the last thread + 1
      template<typename T>
      void transformBlock(int nChunks, int m, const Chunk<T>* pdataIn, long long ldaIn, Chunk<T>* pdataResult, int ldaResult, int resultOffset, int colBlockSize = 15, int firstThread = 0, int lastThread = 0, LatchArray3* bossLatches = nullptr, LatchArray3* workerLatches = nullptr)
      {
        const int mChunks = (m-1) / Chunk<T>::size + 1;
        // we need enough buffer space because we store some additional vectors in it...
        if( pdataIn == pdataResult )
        {
          // in-place
          assert(resultOffset == nChunks);
          assert(ldaResult >= mChunks + nChunks + 2);
        }
        else
        {
          // out-of-place
          assert(resultOffset >= nChunks + 2);
          assert(ldaResult >= mChunks + resultOffset);
          pdataResult = pdataResult + resultOffset - nChunks;
        }

        // this is an approach for hierarchical blocking of
        // for(int i = 0; i < m; i++)
        //   calc i
        //   for(int j = i+1; j < m; j++)
        //     apply i to j

        const int iThread = omp_get_thread_num();
        const int bs = colBlockSize;

        const auto tree_apply = [&](const auto& tree_apply, int beginCol, int endCol, int applyBeginCol, int applyEndCol) -> void
        {
          const int nCol = endCol - beginCol;

          if( nCol < 2*bs )
          {
            if (firstThread == lastThread)
            {
              transformBlock_apply(nChunks, m, pdataIn, ldaIn, pdataResult, ldaResult, resultOffset, beginCol, endCol, applyBeginCol, applyEndCol);
            }
            else
            {
              const int nApplyCol = applyEndCol - applyBeginCol;
              const int nThreads = lastThread - firstThread;
              const int relThreadId = iThread - firstThread;
              auto [localApplyBeginCol, localApplyEndCol] = internal::parallel::distribute(nApplyCol, {relThreadId, nThreads});
              localApplyBeginCol += applyBeginCol;
              localApplyEndCol += applyBeginCol + 1;

              transformBlock_apply(nChunks, m, pdataIn, ldaIn, pdataResult, ldaResult, resultOffset, beginCol, endCol, localApplyBeginCol, localApplyEndCol);
            }
          }
          else
          {
            int middle = beginCol + (nCol/2/bs)*bs;
            tree_apply(tree_apply, beginCol, middle, applyBeginCol, applyEndCol);
            tree_apply(tree_apply, middle, endCol, applyBeginCol, applyEndCol);
          }
        };

        // counter to select current latch for synchronization
        unsigned int iterationCounter = 0;

        const auto tree_calc = [&](const auto& tree_calc, int beginCol, int endCol) -> void
        {
          int nCol = endCol - beginCol;
          if( nCol < 2*bs )
          {
            if (iThread == firstThread || firstThread == lastThread)
            {
              for(int col = beginCol; col < endCol; col++)
              {
                transformBlock_calc(nChunks, m, pdataIn, ldaIn, pdataResult, ldaResult, resultOffset, col);
                transformBlock_apply(nChunks, m, pdataIn, ldaIn, pdataResult, ldaResult, resultOffset, col, col+1, col+1, endCol);
              }
            }
          }
          else
          {
            int middle = beginCol + (nCol/2/bs)*bs;

            tree_calc(tree_calc, beginCol, middle);

            // wait for boss thread to finish...
            const auto cnt = iterationCounter++;
            if( bossLatches )
            {
              if( iThread == firstThread )
                (*bossLatches)[cnt%3].count_down();
              else
                (*bossLatches)[cnt%3].wait();

            }

            tree_apply(tree_apply, beginCol, middle, middle, endCol);

            // wait for all worker threads to finish and reset old currently unused latch
            if( workerLatches )
            {
              if( iThread != firstThread )
                (*workerLatches)[cnt%3].count_down();
              
              if( iThread == firstThread )
              {
                (*workerLatches)[cnt%3].wait();
                resetLatch((*bossLatches)[(cnt+2)%3], 1);
                const auto nThreads = lastThread - firstThread;
                resetLatch((*workerLatches)[(cnt+2)%3], nThreads-1);
              }
            }

            tree_calc(tree_calc, middle, endCol);
          }
        };

        tree_calc(tree_calc,0,m);
      }

      //! internal helper function for transformBlock
      template<typename T>
      [[gnu::always_inline]]
      inline void transformBlock_calc(int nChunks, int m, const Chunk<T>* pdataIn, long long ldaIn, Chunk<T>* pdataResult, int ldaResult, [[maybe_unused]] int resultOffset, int col)
      {
        const int mChunks = (m-1) / Chunk<T>::size + 1;

        const Chunk<T>* pdata = nullptr;
        long long lda = 0;
        if( col == 0 )
        {
          pdata = pdataIn;
          lda = ldaIn;
        }
        else
        {
          pdata = pdataResult;
          lda = ldaResult;
        }

        int firstRow = col / Chunk<T>::size;
        int idx = col % Chunk<T>::size;
        Chunk<T> pivotChunk;
        masked_load_after(pdata[firstRow+lda*col], idx, pivotChunk);
        // Householder projection P = I - 2 v v^T
        // u = x - alpha e_1 with alpha = +- ||x||
        // v = u / ||u||
        T pivot = pdata[firstRow+lda*col][idx];
        Chunk<T> uTu;
        mul(pivotChunk, pivotChunk, uTu);
        {
          int i = firstRow+1;
          for(; i < nChunks; i++)
            fmadd(pdata[i+lda*col], pdata[i+lda*col], uTu);
          for(; i <= nChunks+firstRow; i++)
            fmadd(pdataResult[i+ldaResult*col], pdataResult[i+ldaResult*col], uTu);
        }

        T uTu_sum = sum(uTu) + std::numeric_limits<T>::min();

        // add another minVal, s.t. the Householder reflection is correctly set up even for zero columns
        // (falls back to I - 2 e1 e1^T in that case)
        using RealType = decltype(std::abs(T(0)));
        static_assert(RealType(0) != std::numeric_limits<RealType>::min());
        T alpha = std::sqrt(uTu_sum + std::numeric_limits<RealType>::min());
        if constexpr ( requires(T x){x > 0;} )
          alpha *= (pivot > 0 ? -1 : 1);
        else
          alpha *= (pivot == T(0) ? T(-1) : -pivot / std::abs(pivot));



        // calculate reflection vector
        Chunk<T> vtmp[nChunks+1];
        if( col+1 < m )
        {
          uTu_sum -= pivot*alpha;
          pivot -= alpha;
          index_bcast(pivotChunk, idx, pivot, pivotChunk);
          T beta = T(1)/std::sqrt(uTu_sum);
          mul(beta, pivotChunk, vtmp[0]);
          int i = firstRow+1;
          for(; i < nChunks; i++)
            mul(beta, pdata[i+lda*col], vtmp[i-firstRow]);
          for(; i <= nChunks+firstRow; i++)
            mul(beta, pdataResult[i+ldaResult*col], vtmp[i-firstRow]);
        }

        // set (known) current column to (*,*,*,alpha,0,0,...)
        Chunk<T> alphaChunk;
        index_bcast(Chunk<T>{}, idx, alpha, alphaChunk);
        masked_store_after(alphaChunk, idx, pdataResult[firstRow+ldaResult*col]);
        if( pdataIn == pdataResult )
        {
          // in-place calculation: result stays at top, only need to set zeros below...
          for(int i = firstRow+1; i < mChunks; i++)
            pdataResult[i+ldaResult*col] = Chunk<T>{};
        }
        else
        {
          // out-of-place calculation: move result to the bottom...
          for(int i = firstRow; i >= 0; i--)
            pdataResult[nChunks+i+ldaResult*col] = pdataResult[i+ldaResult*col];
        }

        if( col + 1 < m )
        {
          // store the current reflection vector in the result buffer (location above/below depending on in-place vs. out-of-place)
          Chunk<T>* v = nullptr;
          if( pdataIn == pdataResult )
            v = &pdataResult[mChunks+ldaResult*col] - firstRow;
          else
            v = &pdataResult[0+ldaResult*col] - firstRow - 2;

          for(int i = firstRow; i <= nChunks+firstRow; i++)
            v[i] = vtmp[i-firstRow];


          // calculate vTw if needed later
          if( col % 2 == 1 )
          {
            Chunk<T>* w = nullptr;
            if( pdataIn == pdataResult )
              w = &pdataResult[mChunks+ldaResult*(col-1)] - firstRow;
            else
              w = &pdataResult[0+ldaResult*(col-1)] - firstRow - 2;

            Chunk<T> vTw{};
            for(int i = firstRow; i <= nChunks+firstRow; i++)
              fmadd(v[i], w[i], vTw);
            bcast_sum(vTw);
            w[nChunks+firstRow+1] = vTw;
          }
        }
      }

      //! internal helper function for transformBlock
      template<typename T>
      [[gnu::always_inline]]
      inline void transformBlock_apply(int nChunks, int m, const Chunk<T>* pdataIn, long long ldaIn,
                                Chunk<T>* pdataResult, int ldaResult, [[maybe_unused]] int resultOffset,
                                int beginCol, int endCol, int applyBeginCol, int applyEndCol)
      {
        const int mChunks = (m-1) / Chunk<T>::size + 1;

        if( endCol <= beginCol )
          return;

        const Chunk<T> *v0 = nullptr;
        if( pdataIn == pdataResult )
          v0 = &pdataResult[mChunks];
        else
          v0 = &pdataResult[0] - 2;

        const Chunk<T>* pdata = nullptr;
        long long lda = 0;

        // reverse ordering (write avoiding)
        int j = applyBeginCol;
        for(; j+2 < applyEndCol; j+=3)
        {
          for(int col = beginCol; col < endCol; col++)
          {
            if( col == 0 || col == 1 )
            {
              pdata = pdataIn;
              lda = ldaIn;
            }
            else
            {
              pdata = pdataResult;
              lda = ldaResult;
            }
            int firstRow = col / Chunk<T>::size;
            const Chunk<T>* v = v0 + ldaResult*col - firstRow;
            if( col % 2 == 1 )
            {
              const Chunk<T>* w = v - ldaResult;
              const Chunk<T> vTw = w[nChunks+firstRow+1];
              applyReflection2<T,3>(nChunks, firstRow, j, w, v, vTw, pdata, lda, pdataResult, ldaResult);
            }
          }
        }
        for(int col = beginCol; col < endCol; col++)
        {
          if( col == 0 || col == 1 )
          {
            pdata = pdataIn;
            lda = ldaIn;
          }
          else
          {
            pdata = pdataResult;
            lda = ldaResult;
          }
          int firstRow = col / Chunk<T>::size;
          const Chunk<T>* v = v0 + ldaResult*col - firstRow;
          if( col % 2 == 1 )
          {
            const Chunk<T>* w = v - ldaResult;
            const Chunk<T> vTw = w[nChunks+firstRow+1];
            if( j+1 < applyEndCol )
              applyReflection2<T,2>(nChunks, firstRow, j, w, v, vTw, pdata, lda, pdataResult, ldaResult);
            else if( j < applyEndCol )
              applyReflection2<T,1>(nChunks, firstRow, j, w, v, vTw, pdata, lda, pdataResult, ldaResult);
          }
        }

        // special handling for last column
        {
          int col = endCol - 1;
          if( col == 0 || col == 1 )
          {
            pdata = pdataIn;
            lda = ldaIn;
          }
          else
          {
            pdata = pdataResult;
            lda = ldaResult;
          }
          int firstRow = col / Chunk<T>::size;
          const Chunk<T>* v = v0 + ldaResult*col - firstRow;

          if( col % 2 == 0 && col+1 >= applyBeginCol && col+1 < applyEndCol )
          {
            //std::cout << "col " << col << "\n";
            applyReflection(nChunks, firstRow, col+1, v, pdata, lda, pdataResult, ldaResult);
          }
        }
      }


      //! Helper function for combining two upper triangular factors in an MPI_Reduce operation
      //!
      //! @param invec      upper triangular part of this process (memory layout + padding see implementation)
      //! @param inoutvec   upper triangular part of the next process (memory layout + padding see implementation)
      //! @param len        number of entries, should always be 1 as we use a dedicated MPI data type
      //! @param datatype   MPI data type, used to extract actual length
      //!
      template<typename T>
      void combineTwoBlocks(const T* invec, T* inoutvec, [[maybe_unused]] const int* len, const MPI_Datatype* datatype)
      {
        assert( *len == 1 );
        int size;
        int ierr = MPI_Type_size(*datatype, &size);
        assert( ierr == MPI_SUCCESS );
        assert( size % sizeof(T) == 0 );
        size /= sizeof(T);

        // cannot easily check that datatype is the type returned by MPI_Type_contigous(size, parallel::mpiType<T>(), &type)
        // assert( *datatype == parallel::mpiType<T>() );

        // get dimensions
        int m = 1, mChunks = 1;
        while( m*mChunks*Chunk<T>::size < size )
        {
          if( m % Chunk<T>::size == 0 )
            mChunks++;
          m++;
        }
        assert( mChunks == (m-1) / Chunk<T>::size + 1 );
        assert( mChunks*Chunk<T>::size*m == size );

        const auto ldaBuff = 2*mChunks+2;

        // get required buffer
        std::unique_ptr<Chunk<T>[]> buff{new Chunk<T>[ldaBuff*m]};

        // check alignement of buffers, we might be lucky often (because MPI allocated aligned buffers or we get our own buffers from the MPI_Allreduce call)
        const Chunk<T>* invecChunked = nullptr;
        Chunk<T>* inoutvecChunked = nullptr;
        if( std::bit_cast<std::uintptr_t>(invec) % ALIGNMENT == 0 )
          invecChunked = (const Chunk<T>*) invec;
        if( std::bit_cast<std::uintptr_t>(inoutvec) % ALIGNMENT == 0 )
          inoutvecChunked = (Chunk<T>*) inoutvec;

        // copy to buffer
        if( invecChunked )
        {
          // aligned variant
          for(int j = 0; j < m; j++)
            for(int i = 0; i < mChunks; i++)
              buff[i+j*ldaBuff] = invecChunked[i+j*mChunks];
        }
        else
        {
          // unaligned variant
          for(int j = 0; j < m; j++)
            for(int i = 0; i < mChunks; i++)
              unaligned_load(invec+(i+j*mChunks)*Chunk<T>::size, buff[i+j*ldaBuff]);
        }
        if( inoutvecChunked )
        {
          // aligned variant
          for(int j = 0; j < m; j++)
            for(int i = 0; i < mChunks; i++)
              buff[mChunks+i+j*ldaBuff] = inoutvecChunked[i+j*mChunks];
        }
        else
        {
          // unaligned variant
          for(int j = 0; j < m; j++)
            for(int i = 0; i < mChunks; i++)
              unaligned_load(inoutvec+(i+j*mChunks)*Chunk<T>::size, buff[mChunks+i+j*ldaBuff]);
        }

        transformBlock(mChunks, m, &buff[0], ldaBuff, &buff[0], ldaBuff, mChunks);

        // copy back to inoutvec
        if( inoutvecChunked )
        {
          // aligned variant
          for(int j = 0; j < m; j++)
            for(int i = 0; i < mChunks; i++)
              inoutvecChunked[i+j*mChunks] = buff[i+j*ldaBuff];
        }
        else
        {
          // unaligned variant
          for(int j = 0; j < m; j++)
            for(int i = 0; i < mChunks; i++)
              unaligned_store(buff[i+j*ldaBuff], inoutvec+(i+j*mChunks)*Chunk<T>::size);
        }
      }

      //! wrapper function for MPI (otherwise the function signature does not match!)
      template<typename T>
      void combineTwoBlocks_mpiOp(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype)
      {
        combineTwoBlocks<T>((const T*)invec, (T*)inoutvec, len, datatype);
      }
    }
  }


  // implement multivector TSQR
  template<typename T>
  void block_TSQR(const MultiVector<T>& M, Tensor2<T>& R, int reductionFactor, bool mpiGlobal, int colBlockingSize)
  {
    // input dimensions
    const long long n = M.rows();
    const long long nTotalChunks = M.rowChunks();
    const int m = M.cols();
    const int mChunks = (m-1) / Chunk<T>::size + 1;
    // get the number of OpenMP threads
    int nMaxThreads = omp_get_max_threads();

    // calculate performance tuning parameters
    {
      // L1 cache size per core (in chunks)
      constexpr int cacheSize_L1 = 32*1024 / (Chunk<T>::size * sizeof(T));
      // L2 cache size per core (in chunks)
      constexpr int cacheSize_L2 = 1*1024*1024 / (Chunk<T>::size * sizeof(T));

      // automatically choose suitable reduction factor
      if( reductionFactor == 0 )
      {
        // max. reductionFactor (for small number of columns, ensure applyReflection2<T,3> fits into L1)
        constexpr int maxReductionFactor = int(0.74 * cacheSize_L1 / (3+2));

        // choose the reduction factor such that 2 blocks of (reductionFactor x M.cols()) fit into the L2 cache
        reductionFactor = std::min(maxReductionFactor, int(0.74 * cacheSize_L2 / M.cols()) );
        reductionFactor = std::min<long long>(reductionFactor, std::max<long long>(reductionFactor/2, nTotalChunks / (2*nMaxThreads)));
        reductionFactor = std::max(1, reductionFactor);
      }

      // automaticall choose suitable colBlockingSize
      if( colBlockingSize == 0 )
      {
        // not sure how to choose this best, should be a multiple of 3 (3-way unrolling with applyReflection2<T,3>)
        // 15 seems to work fine...
        colBlockingSize = 12;
      }
    }

    // calculate dimensions and block sizes
    const int nChunks = reductionFactor;
    const int ldaBuff = std::max(nChunks, mChunks) + mChunks + 2;
    const int nBuffer = m*ldaBuff;
    // index to the next free block in plocalBuff
    const int localBuffOffset = std::max(nChunks, mChunks) + 2;
//printf("nBuffer: %d\n", nBuffer);
    const long long nIter = (nTotalChunks-1) / nChunks + 1;
    const long long lda = M.colStrideChunks();

    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"rows", "cols", "reductionFactor", "colBlockingSize"},{n, m, reductionFactor, colBlockingSize}}, // arguments
        {{(1.+1./reductionFactor)*(n*(m + m*(m-1.)))*kernel_info::FMA<T>()}, // flops - roughly estimated
         {(n*m)*kernel_info::Load<T>() + (m*m)*kernel_info::Store<T>()}} // data transfers
        );

    // consider empty matrices
    if( M.cols() == 0 )
    {
      R.resize(M.cols(), M.cols());
      return;
    }

    // reduce #threads if there is not enough work to do...
    //int nDesiredThreads = std::min<long long>((nIter-1)/2+1, nMaxThreads);

    std::unique_ptr<Chunk<T>[]> pThread0Buff;
    std::unique_ptr<Chunk<T>[]> presultBuff(new Chunk<T>[mChunks*m]);

// avoid omp parallel with the num_threads-argument because it might stop a running thread that needs to respawn later (GCC 10 behavior in some cases?).
//#pragma omp parallel num_threads(nDesiredThreads)
//
// Simple reduce the number of threads later instead...
//

    constexpr int falseSharingStride = 16;
    std::vector<Chunk<T>*> plocalBuff_allThreads(falseSharingStride*nMaxThreads); // take care to index into with proper offset!

    std::unique_ptr<internal::LatchArray3[]> bossLatchBuff(new internal::LatchArray3[nMaxThreads*2]);
    std::unique_ptr<internal::LatchArray3[]> workerLatchBuff(new internal::LatchArray3[nMaxThreads*2]);
    std::array<internal::LatchArray3*,2> localBossLatches = {&bossLatchBuff[0], &bossLatchBuff[nMaxThreads]};
    std::array<internal::LatchArray3*,2> localWorkerLatches = {&workerLatchBuff[0], &workerLatchBuff[nMaxThreads]};

#pragma omp parallel
    {
      auto [iThread,nThreads] = internal::parallel::ompThreadInfo();
      // only work on a subset of threads if the input dimension is too small
      if( nIter < nThreads*2 )
        nThreads = 1 + (nIter-1)/2;

      std::unique_ptr<Chunk<T>[]> plocalBuff;
      if( iThread < nThreads )
      {
        plocalBuff.reset(new Chunk<T>[nBuffer]);
        plocalBuff_allThreads[falseSharingStride*iThread] = plocalBuff.get();

        // fill with zero
        for(int j = 0; j < m; j++)
          for(int i = 0; i < mChunks; i++)
            plocalBuff[localBuffOffset + i + j*ldaBuff] = Chunk<T>{};

        const auto& [firstIter, lastIter] = internal::parallel::distribute(nIter, {iThread, nThreads});

        for(long long iter = firstIter; iter <= lastIter; iter++)
        {
          const int nRemainingChunks = nTotalChunks-iter*nChunks;
          internal::HouseholderQR::transformBlock(std::min(nChunks, nRemainingChunks), m, &M.chunk(nChunks*iter,0), lda, &plocalBuff[0], ldaBuff, localBuffOffset, colBlockingSize);
        }
      }

      // tree reduction over threads
      bool wasBossThread = false; // keep track of last iteration's boss threads
      int cnt = 1;
      for(int nextThread = 1; nextThread < nThreads; nextThread*=2, cnt++)
      {
        int bossThread, lastThread; // first (including) and last (excluding) threads in the team
        if (iThread < nThreads)
        {
          const int threadteamSize = nextThread*2;
          bossThread = iThread - iThread % threadteamSize;
          lastThread = std::min(bossThread + threadteamSize, nThreads);
          // in following special case, could add following unused threads
          //if (lastThread < nThreads && bossThread+3*nextThread >= nThreads)
          //    lastThread = nThreads;

          // create this iteration's local latches (before global barrier to ensure all threads in team have the same view of them)
          if (iThread == bossThread)
          {
            resetLatch(localBossLatches[cnt%2][bossThread][0], 1);
            resetLatch(localBossLatches[cnt%2][bossThread][1], 1);
            resetLatch(localBossLatches[cnt%2][bossThread][2], 1);
            resetLatch(localWorkerLatches[cnt%2][bossThread][0], lastThread-bossThread - 1);
            resetLatch(localWorkerLatches[cnt%2][bossThread][1], lastThread-bossThread - 1);
            resetLatch(localWorkerLatches[cnt%2][bossThread][2], lastThread-bossThread - 1);
          }
        }
#pragma omp barrier
        if (iThread < nThreads)
        {
          if (bossThread+nextThread < nThreads)
          {
            const auto bossLocalBuff = &plocalBuff_allThreads[falseSharingStride*bossThread][0];
            const auto otherLocalBuff = &plocalBuff_allThreads[falseSharingStride*(bossThread+nextThread)][localBuffOffset];
            internal::HouseholderQR::transformBlock(mChunks, m, otherLocalBuff, ldaBuff, bossLocalBuff, ldaBuff, localBuffOffset, colBlockingSize, bossThread, lastThread, &localBossLatches[cnt%2][bossThread], &localWorkerLatches[cnt%2][bossThread]);
          }

          // destruct previous iteration's local barriers (we wait for one iteration to ensure that there is a global barrier inbetween last use and destruction of the barrier)
          // the very last iteration's local barriers are destructed after the loop and after another global barrier
          // remark: explicit destruction only needed if the destructor has side effects
          // remember this iterations boss threads (in order to correctly destruct barriers in next iteration)
          wasBossThread = (iThread == bossThread);
        }
      }

      if( iThread == 0 )
        pThread0Buff = std::move(plocalBuff);

      // wait for other threads so the otherLocalBuff pointers and local barriers are valid until all threads have finished
#pragma omp barrier
    }

    if( mpiGlobal )
    {
      const auto& [iProc,nProcs] = internal::parallel::mpiProcInfo();
      if( nProcs > 1 )
      {
        // compress result
        for(int j = 0; j < m; j++)
          for(int i = 0; i < mChunks; i++)
            presultBuff[i+mChunks*j] = pThread0Buff[localBuffOffset + i+ldaBuff*j];

//#define MPI_TIMINGS
#ifdef MPI_TIMINGS
        double wtime = omp_get_wtime();
#endif
        // define a dedicated datatype as MPI reduction operations are defined element-wise
        MPI_Datatype tsqrType;
        if( MPI_Type_contiguous(mChunks*Chunk<T>::size*m, internal::parallel::mpiType<T>(), &tsqrType) != MPI_SUCCESS )
          throw std::runtime_error("Failure returned from MPI_Type_contiguous");

        if( MPI_Type_commit(&tsqrType) != MPI_SUCCESS )
          throw std::runtime_error("Failure returned from MPI_Type_commit");

        // register MPI reduction operation, currently commutative - not sure if good for reproducibility...
        MPI_Op tsqrOp;
        if( MPI_Op_create(&internal::HouseholderQR::combineTwoBlocks_mpiOp<T>, 1, &tsqrOp) != MPI_SUCCESS )
          throw std::runtime_error("Failure returned from MPI_Op_create");

        // actual MPI reduction, reusing buffers
        std::swap(pThread0Buff, presultBuff);
        if( MPI_Allreduce(pThread0Buff.get(), presultBuff.get(), 1, tsqrType, tsqrOp, MPI_COMM_WORLD) != MPI_SUCCESS )
          throw std::runtime_error("Failure returned from MPI_Allreduce");

        // unregister MPI reduction operation
        if( MPI_Op_free(&tsqrOp) != MPI_SUCCESS )
          throw std::runtime_error("Failure returned from MPI_Op_free");

        // free dedicated MPI type
        if( MPI_Type_free(&tsqrType) != MPI_SUCCESS )
          throw std::runtime_error("Failure returned from MPI_Type_free");

#ifdef MPI_TIMINGS
        wtime = omp_get_wtime() - wtime;
        if( iProc == 0 )
          std::cout << "TSQR MPI wtime: " << wtime << "\n";
#endif

        // copy result to R
        R.resize(m,m);
        for(int j = 0; j < m; j++)
          for(int i = 0; i < m; i++)
            R(i,j) = presultBuff[ i/Chunk<T>::size + mChunks*j ][ i%Chunk<T>::size ];

        return;
      }
    }

    // not handled through MPI
    // copy result to R
    R.resize(m,m);
    for(int j = 0; j < m; j++)
      for(int i = 0; i < m; i++)
        R(i,j) = pThread0Buff[ localBuffOffset + i/Chunk<T>::size + ldaBuff*j ][ i%Chunk<T>::size ];
  }
}


#endif // PITTS_MULTIVECTOR_TSQR_IMPL_HPP
