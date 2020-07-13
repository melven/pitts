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
#include <memory>
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! helper namespace for TSQR routines
  namespace HouseholderQR
  {
    //! internal details of the block TSQR implementation
    namespace internal
    {
      //! Apply a Householder rotation of the form (I - v v^T) with ||v|| = sqrt(2)
      //!
      //! Usually, a Householder rotation has the form (I - 2 v v^T), the factor 2 is included in v to improve the performance.
      //!
      //! @tparam T           underlying data type
      //!
      //! @param nChunks      number of rows divided by the Chunk size
      //! @param firstRow     index of the chunk that contains the current pivot element (0 <= firstRow < nChunks)
      //! @param col          current column index
      //! @param v            Householder vector with norm sqrt(2), the upper firstRow-1 chunks are ignored.
      //! @param pdata        column-major input array with dimension nChunks*lda; can be identical to pdataResult for in-place calculation
      //! @param lda          offset of columns in pdata
      //! @param pdataResult  dense, column-major output array with dimension nChunks*#columns, the upper firstRow-1 chunks of rows are not touched
      //!
      template<typename T>
      void applyHouseholderRotation(int nChunks, int firstRow, int col, const Chunk<T>* v, const Chunk<T>* pdata, int lda, Chunk<T>* pdataResult)
      {
        Chunk<T> vTx{};
        {
          int i = firstRow;
          Chunk<T> vTx_{};
          for(; i+1 < nChunks; i+=2)
          {
            fmadd(v[i], pdata[i+lda*col], vTx);
            fmadd(v[i+1], pdata[i+1+lda*col], vTx_);
          }
          fmadd(1., vTx_, vTx);
          for(; i < nChunks; i++)
            fmadd(v[i], pdata[i+lda*col], vTx);
        }
        bcast_sum(vTx);
        for(int i = firstRow; i < nChunks; i++)
          fnmadd(vTx, v[i], pdata[i+lda*col], pdataResult[i+nChunks*col]);
      }


      //! Apply two consecutive Householder rotations of the form (I - v v^T) (I - w w^T) with ||v|| = ||w|| = sqrt(2)
      //!
      //! Usually, a Householder rotation has the form (I - 2 v v^T), the factor 2 is included in v and w to improve the performance.
      //!
      //! This routine has the same effect as calling applyHouseholderRotation twice, first with vector w and then with vector v.
      //! Using this routine avoids to transfer required colums to/from the cache twice.
      //!
      //! Exploits (I - v v^T) (I -w w^T) = I - v (v^T - v^T w w^T) - w w^T where v^T w can be calculated in advance.
      //!
      //! @tparam T           underlying data type
      //!
      //! @param nChunks      number of rows divided by the Chunk size
      //! @param firstRow     index of the chunk that contains the current pivot element (0 <= firstRow < nChunks)
      //! @param col          current column index
      //! @param w            first Householder vector with norm sqrt(2), the upper firstRow-1 chunks are ignored.
      //! @param v            second Householder vector with norm sqrt(2), the upper firstRow-1 chunks are ignored.
      //! @param vTw          scalar product of v and w; required as we apply both transformations at once
      //! @param pdata        column-major input array with dimension nChunks*lda; can be identical to pdataResult for in-place calculation
      //! @param lda          offset of columns in pdata
      //! @param pdataResult  dense, column-major output array with dimension nChunks*#columns, the upper firstRow-1 chunks of rows are not touched
      //!
      template<typename T>
      void applyHouseholderRotation2(int nChunks, int firstRow, int j, const Chunk<T>* w, const Chunk<T>* v, const Chunk<T> &vTw, const Chunk<T>* pdata, int lda, Chunk<T>* pdataResult)
      {
        Chunk<T> wTx{};
        Chunk<T> vTx{};
        for(int i = firstRow; i < nChunks; i++)
        {
          fmadd(w[i], pdata[i+lda*j], wTx);
          fmadd(v[i], pdata[i+lda*j], vTx);
        }
        bcast_sum(wTx);
        bcast_sum(vTx);
        fnmadd(vTw, wTx, vTx);
        for(int i = firstRow; i < nChunks; i++)
        {
          fnmadd(wTx, w[i], pdata[i+lda*j], pdataResult[i+nChunks*j]);
          fnmadd(vTx, v[i], pdataResult[i+nChunks*j]);
        }
      }


      //! Apply two consecutive Householder rotations of the form (I - v v^T) (I - w w^T) with ||v|| = ||w|| = sqrt(2) to two columns
      //!
      //! Usually, a Householder rotation has the form (I - 2 v v^T), the factor 2 is included in v and w to improve the performance.
      //!
      //! This routine has the same effect as calling apply2HouseholderRotations twice, once for column col, once for column col+1
      //! Using this routine avoids to transfer required Householder vectors from the cache twice.
      //!
      //! Exploits (I - v v^T) (I -w w^T) = I - v (v^T - v^T w w^T) - w w^T where v^T w can be calculated in advance.
      //!
      //! @tparam T           underlying data type
      //!
      //! @param nChunks      number of rows divided by the Chunk size
      //! @param firstRow     index of the chunk that contains the current pivot element (0 <= firstRow < nChunks)
      //! @param col          index of the first of the two consecutive columns.
      //! @param w            first Householder vector with norm sqrt(2), the upper firstRow-1 chunks are ignored.
      //! @param v            second Householder vector with norm sqrt(2), the upper firstRow-1 chunks are ignored.
      //! @param vTw          scalar product of v and w; required as we apply both transformations at once
      //! @param pdata        column-major input array with dimension nChunks*lda; can be identical to pdataResult for in-place calculation
      //! @param lda          offset of columns in pdata
      //! @param pdataResult  dense, column-major output array with dimension nChunks*#columns, the upper firstRow-1 chunks of rows are not touched
      //!
      template<typename T>
      void applyHouseholderRotation2x2(int nChunks, int firstRow, int j, const Chunk<T>* w, const Chunk<T>* v, const Chunk<T> &vTw, const Chunk<T>* pdata, int lda, Chunk<T>* pdataResult)
      {
        Chunk<T> wTx{};
        Chunk<T> vTx{};
        Chunk<T> wTy{};
        Chunk<T> vTy{};
        for(int i = firstRow; i < nChunks; i++)
        {
          fmadd(w[i], pdata[i+lda*j], wTx);
          fmadd(v[i], pdata[i+lda*j], vTx);
          fmadd(w[i], pdata[i+lda*(j+1)], wTy);
          fmadd(v[i], pdata[i+lda*(j+1)], vTy);
        }
        bcast_sum(wTx);
        bcast_sum(vTx);
        fnmadd(vTw, wTx, vTx);
        bcast_sum(wTy);
        bcast_sum(vTy);
        fnmadd(vTw, wTy, vTy);
        for(int i = firstRow; i < nChunks; i++)
        {
          fnmadd(wTx, w[i], pdata[i+lda*j], pdataResult[i+nChunks*j]);
          fnmadd(vTx, v[i], pdataResult[i+nChunks*j]);
          fnmadd(wTy, w[i], pdata[i+lda*(j+1)], pdataResult[i+nChunks*(j+1)]);
          fnmadd(vTy, v[i], pdataResult[i+nChunks*(j+1)]);
        }
      }

    }
  }
}


#endif // PITTS_MULTIVECTOR_TSQR_HPP
