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
      //! Usually, a Householder rotation has the form (I - 2 v v^T), the factor 2 is included in v to improve the performance
      //!
      //! @tparam T           underlying data type
      //!
      //! @param nChunks      number of rows divided by the Chunk size
      //! @param firstRow     index of the chunk that contains the current pivot element (0 <= firstRow < nChunks)
      //! @param col          current column index
      //! @param v            vector v with norm sqrt(2), the upper firstRow-1 chunks are ignored.
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
    }
  }
}


#endif // PITTS_MULTIVECTOR_TSQR_HPP
