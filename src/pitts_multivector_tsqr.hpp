/*! @file pitts_multivector_tsqr.hpp
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
#ifndef PITTS_MULTIVECTOR_TSQR_HPP
#define PITTS_MULTIVECTOR_TSQR_HPP

// includes
#include "pitts_parallel.hpp"
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"
#include <cassert>
#include <memory>
#include <cstdint>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
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

        Chunk<T> wTx[NC]{};
        Chunk<T> vTx[NC]{};
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
            for(int j = 0; j < NC; j++)
            {
              Chunk<T> tmp;
              fnmadd(wTx[j], w[i], pdataResult[i+ldaResult*(col+j)], tmp);
              fnmadd(vTx[j], v[i], tmp, pdataResult[i+ldaResult*(col+j)]);
            }
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
            for(int j = 0; j < NC; j++)
            {
              Chunk<T> tmp;
              fnmadd(wTx[j], w[i], pdata[i+lda*(col+j)], tmp);
              fnmadd(vTx[j], v[i], tmp, pdataResult[i+ldaResult*(col+j)]);
            }
          }
          for(int i = nChunks; i <= nChunks+firstRow; i++)
          {
            Chunk<T> tmp;
            for(int j = 0; j < NC; j++)
            {
              fnmadd(wTx[j], w[i], pdataResult[i+ldaResult*(col+j)], tmp);
              fnmadd(vTx[j], v[i], tmp, pdataResult[i+ldaResult*(col+j)]);
            }
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

        const std::function<void(int,int,int,int)> tree_apply = [&](int beginCol, int endCol, int applyBeginCol, int applyEndCol)
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
            tree_apply(beginCol, middle, applyBeginCol, applyEndCol);
            tree_apply(middle, endCol, applyBeginCol, applyEndCol);
          }
        };

        const std::function<void(int,int)> tree_calc = [&](int beginCol, int endCol)
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
            tree_calc(beginCol, middle);
            tree_apply(beginCol, middle, middle, endCol);
            tree_calc(middle, endCol);
          }
        };

        tree_calc(0,m);
      }

      //! internal helper function for transformBlock
      template<typename T>
      [[gnu::always_inline]]
      inline void transformBlock_calc(int nChunks, int m, const Chunk<T>* pdataIn, long long ldaIn, Chunk<T>* pdataResult, int ldaResult, int resultOffset, int col)
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
        Chunk<T> uTu{};
        fmadd(pivotChunk, pivotChunk, uTu);
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
        T alpha = std::sqrt(uTu_sum + std::numeric_limits<T>::min());
        //alpha *= (pivot == 0 ? -1. : -pivot / std::abs(pivot));
        alpha *= (pivot > 0 ? -1 : 1);


        // calculate reflection vector
        Chunk<T> vtmp[nChunks+1];
        if( col+1 < m )
        {
          uTu_sum -= pivot*alpha;
          pivot -= alpha;
          index_bcast(pivotChunk, idx, pivot, pivotChunk);
          T beta = 1/std::sqrt(uTu_sum);
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
                                Chunk<T>* pdataResult, int ldaResult, int resultOffset,
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
      //! @param len        number of entries (including all padding, etc)
      //! @param datatype   MPI data type (ignored)
      //!
      template<typename T>
      void combineTwoBlocks(const T* invec, T* inoutvec, const int* len, const MPI_Datatype* datatype)
      {
        assert( *datatype == parallel::mpiType<T>() );

        // get dimensions
        int m = 1, mChunks = 1;
        while( m*mChunks*Chunk<T>::size < *len )
        {
          if( m % Chunk<T>::size == 0 )
            mChunks++;
          m++;
        }
        assert( mChunks == (m-1) / Chunk<T>::size + 1 );
        assert( mChunks*Chunk<T>::size*m == *len );

        const auto ldaBuff = 2*mChunks+2;

        // get required buffer
        std::unique_ptr<Chunk<T>[]> buff{new Chunk<T>[ldaBuff*m]};

        // check alignement of buffers, we might be lucky often (because MPI allocated aligned buffers or we get our own buffers from the MPI_Allreduce call)
        const Chunk<T>* invecChunked = nullptr;
        Chunk<T>* inoutvecChunked = nullptr;
        if( reinterpret_cast<std::uintptr_t>(invec) % ALIGNMENT == 0 )
          invecChunked = (const Chunk<T>*) invec;
        if( reinterpret_cast<std::uintptr_t>(inoutvec) % ALIGNMENT == 0 )
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


  //! Calculate R from a QR decomposition of a multivector M
  //!
  //! MPI+OpenMP parallel TSQR implementation that calculates ony R. Q is built implicitly but never stored to reduce memory transfers.
  //! It is based on Householder reflections, robust and rank-preserving (just returns rank-deficient R if M does not have full rank).
  //!
  //! @tparam T underlying data type
  //!
  //! @param M                input matrix, possibly distributed over multiple MPI processes
  //! @param R                output matrix R of a QR decomposition of M (with the same singular values and right-singular vectors as M)
  //! @param reductionFactor  (performance-tuning parameter) defines the #chunks of the work-array in the TSQR reduction;
  //!                         set to zero to let this function choose a suitable value automatically
  //! @param mpiGlobal        perform a reduction of R over all MPI processes? (if false, each MPI process does its individual QR decomposition)
  //! @param colBlockingSize  (performance-tuning parameter) used for improving data accesses when m is large
  //!                         set to zero to let this function choose a suitable value automatically
  //!
  template<typename T>
  void block_TSQR(const MultiVector<T>& M, Tensor2<T>& R, int reductionFactor = 0, bool mpiGlobal = true, int colBlockingSize = 0)
  {
    // input dimensions
    const long long n = M.rows();
    const int m = M.cols();
    const int mChunks = (m-1) / Chunk<T>::size + 1;

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
    const int ldaBuff = nChunks + mChunks+2;
    const int nBuffer = m*ldaBuff;
//printf("nBuffer: %d\n", nBuffer);
    const long long nTotalChunks = M.rowChunks();
    const long long nIter = nTotalChunks / nChunks;
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

    // get the number of OpenMP threads
    int nMaxThreads = omp_get_max_threads();
    // reduce #threads if there is not enough work to do...
    //int nDesiredThreads = std::min<long long>((nIter-1)/2+1, nMaxThreads);

    std::unique_ptr<Chunk<T>[]> psharedBuff(new Chunk<T>[(mChunks*nMaxThreads+2)*m]);
    std::unique_ptr<Chunk<T>[]> presultBuff(new Chunk<T>[mChunks*m]);

// avoid omp parallel with the num_threads-argument because it might stop a running thread that needs to respawn later (GCC 10 behavior in some cases?).
//#pragma omp parallel num_threads(nDesiredThreads)
//
// Simple reduce the number of threads later instead...
//

#pragma omp parallel if(nMaxThreads < 2*nIter)
    {
      const auto& [iThread,nThreads] = internal::parallel::ompThreadInfo();

      std::unique_ptr<Chunk<T>[]> plocalBuff{new Chunk<T>[nBuffer]};

      // fill with zero
      for(int i = 0; i < nBuffer; i++)
          plocalBuff[i] = Chunk<T>{};

      // index to the next free block in plocalBuff
      int localBuffOffset = nChunks+2;

#pragma omp for schedule(static)
      for(long long iter = 0; iter < nIter; iter++)
      {
        internal::HouseholderQR::transformBlock(nChunks, m, &M.chunk(nChunks*iter,0), lda, &plocalBuff[0], ldaBuff, localBuffOffset, colBlockingSize);
      }
      // remainder (missing bottom part that is smaller than nChunk*Chunk::size rows
      if( iThread == nThreads-1 && nIter*nChunks < nTotalChunks )
      {
        const int nLastChunks = nTotalChunks-nIter*nChunks;
        internal::HouseholderQR::transformBlock(nLastChunks, m, &M.chunk(nChunks*nIter,0), lda, &plocalBuff[0], ldaBuff, localBuffOffset, colBlockingSize);
      }

      if( nThreads == 1 )
      {
        // compress result
        for(int j = 0; j < m; j++)
          for(int i = 0; i < mChunks; i++)
            presultBuff[i+mChunks*j] = plocalBuff[localBuffOffset + i + ldaBuff*j];
      }
      else
      {
        const int offset = iThread*mChunks;
        const int ldaSharedBuff = nThreads*mChunks+2;
        for(int j = 0; j < m; j++)
          for(int i = 0; i < mChunks; i++)
            psharedBuff[offset + i + ldaSharedBuff*j] = plocalBuff[localBuffOffset + i + ldaBuff*j];

#pragma omp barrier

#pragma omp master
        {
          // reduce shared buffer
          internal::HouseholderQR::transformBlock((nThreads-1)*mChunks, m, &psharedBuff[0], ldaSharedBuff, &psharedBuff[0], ldaSharedBuff, (nThreads-1)*mChunks, colBlockingSize);

          // compress result
          for(int j = 0; j < m; j++)
            for(int i = 0; i < mChunks; i++)
              presultBuff[i+mChunks*j] = psharedBuff[i+ldaSharedBuff*j];
        }
      }
    }

    if( mpiGlobal )
    {
      const auto& [iProc,nProcs] = internal::parallel::mpiProcInfo();
      if( nProcs > 1 )
      {
        // register MPI reduction operation, currently commutative - not sure if good for reproducibility...
        // we should use a dedicated data type to prevent MPI from possibly splitting up the messages...
        MPI_Op tsqrOp;
        if( MPI_Op_create(&internal::HouseholderQR::combineTwoBlocks_mpiOp<T>, 1, &tsqrOp) != MPI_SUCCESS )
          throw std::runtime_error("Failure returned from MPI_Op_create");

        // actual MPI reduction, reusing buffers
        std::swap(psharedBuff, presultBuff);
        if( MPI_Allreduce(psharedBuff.get(), presultBuff.get(), mChunks*Chunk<T>::size*m, internal::parallel::mpiType<T>(), tsqrOp, MPI_COMM_WORLD) != MPI_SUCCESS )
          throw std::runtime_error("Failure returned from MPI_Allreduce");

        // unregister MPI reduction operation
        if( MPI_Op_free(&tsqrOp) != MPI_SUCCESS )
          throw std::runtime_error("Failure returned from MPI_Op_free");
      }
    }

    // copy result to R
    R.resize(m,m);
    for(int j = 0; j < m; j++)
      for(int i = 0; i < m; i++)
        R(i,j) = presultBuff[ i/Chunk<T>::size + mChunks*j ][ i%Chunk<T>::size ];
  }
}


#endif // PITTS_MULTIVECTOR_TSQR_HPP
