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
#include <omp.h>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! helper namespace for TSQR routines
    namespace HouseholderQR
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
      //! @param pdata        column-major input array with dimension lda*#columns; can be identical to pdataResult for in-place calculation
      //! @param lda          offset of columns in pdata
      //! @param pdataResult  dense, column-major output array with dimension nChunks*#columns, the upper firstRow-1 chunks of rows are not touched
      //!
      template<typename T>
      inline void applyRotation(int nChunks, int firstRow, int col, const Chunk<T>* v, const Chunk<T>* pdata, long long lda, Chunk<T>* pdataResult)
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
          fmadd(T(1), vTx_, vTx);
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
      //! This routine has the same effect as calling applyRotation twice, first with vector w and then with vector v.
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
      //! @param pdata        column-major input array with dimension lda*#columns; can be identical to pdataResult for in-place calculation
      //! @param lda          offset of columns in pdata
      //! @param pdataResult  dense, column-major output array with dimension nChunks*#columns, the upper firstRow-1 chunks of rows are not touched
      //!
      template<typename T>
      inline void applyRotation2(int nChunks, int firstRow, int j, const Chunk<T>* w, const Chunk<T>* v, const Chunk<T> &vTw, const Chunk<T>* pdata, long long lda, Chunk<T>* pdataResult)
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
      //! This routine has the same effect as calling applyRotation2 twice, once for column col, once for column col+1
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
      //! @param pdata        column-major input array with dimension lda*#columns; can be identical to pdataResult for in-place calculation
      //! @param lda          offset of columns in pdata
      //! @param pdataResult  dense, column-major output array with dimension nChunks*#columns, the upper firstRow-1 chunks of rows are not touched
      //!
      template<typename T>
      inline void applyRotation2x2(int nChunks, int firstRow, int j, const Chunk<T>* w, const Chunk<T>* v, const Chunk<T> &vTw, const Chunk<T>* pdata, long long lda, Chunk<T>* pdataResult)
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


      //! Calculate the upper triangular part R from a QR-decomposition of a small rectangular block (with more rows than columns)
      //!
      //! Can work in-place or out-of-place.
      //!
      //! @tparam T           underlying data type
      //!
      //! @param nChunks      number of rows divided by the Chunk size
      //! @param m            number of columns
      //! @param pdataIn      column-major input array with dimension ldaIn*m; can be identical to pdataResult for in-place calculation
      //! @param ldaIn        offset of columns in pdata
      //! @param pdataResult  dense, column-major output array with dimension nChunks*m; contains the upper triangular R on exit; the lower triangular part is set to zero
      //!
      template<typename T>
      void transformBlock(int nChunks, int m, const Chunk<T>* pdataIn, long long ldaIn, Chunk<T>* pdataResult)
      {
        int nPadded = nChunks*Chunk<T>::size;
        Chunk<T> buff_v[nChunks];
        Chunk<T> buff_w[nChunks];
        Chunk<T>* v = buff_v;
        Chunk<T>* w = buff_w;
        const Chunk<T>* pdata = pdataIn;
        long long lda = ldaIn;
        for(int col = 0; col < std::min(m, nPadded); col++)
        {
          int firstRow = col / Chunk<T>::size;
          int idx = col % Chunk<T>::size;
          Chunk<T> pivotChunk;
          masked_load_after(pdata[firstRow+lda*col], idx, pivotChunk);
          // Householder projection P = I - 2 v v^T
          // u = x - alpha e_1 with alpha = +- ||x||
          // v = u / ||u||
          double pivot = pdata[firstRow+lda*col][idx];
          Chunk<T> uTu{};
          fmadd(pivotChunk, pivotChunk, uTu);
          for(int i = firstRow+1; i < nChunks; i++)
            fmadd(pdata[i+lda*col], pdata[i+lda*col], uTu);

          T uTu_sum = sum(uTu) + std::numeric_limits<double>::min();

          // add another minVal, s.t. the Householder reflection is correctly set up even for zero columns
          // (falls back to I - 2 e1 e1^T in that case)
          T alpha = std::sqrt(uTu_sum + std::numeric_limits<double>::min());
          //alpha *= (pivot == 0 ? -1. : -pivot / std::abs(pivot));
          alpha *= (pivot > 0 ? -1 : 1);

          uTu_sum -= pivot*alpha;
          pivot -= alpha;
          Chunk<T> alphaChunk;
          index_bcast(Chunk<T>{}, idx, alpha, alphaChunk);
          if( col+1 < m )
          {
            T beta = 1/std::sqrt(uTu_sum);
            fmadd(T(-1), alphaChunk, pivotChunk);
            mul(beta, pivotChunk, v[firstRow]);
            for(int i = firstRow+1; i < nChunks; i++)
              mul(beta, pdata[i+lda*col], v[i]);
          }

          // apply I - 2 v v^T     (the factor 2 is already included in v)
          // we already know column col
          masked_store_after(alphaChunk, idx, pdataResult[firstRow+nChunks*col]);
          for(int i = firstRow+1; i < nChunks; i++)
            pdataResult[i+nChunks*col] = Chunk<T>{};

          // outer loop unroll (v and previous v in w)
          if( col % 2 == 1 )
          {
            if( col == 1 )
            {
              pdata = pdataIn;
              lda = ldaIn;
            }

            // (I-vv^T)(I-ww^T) = I - vv^T - ww^T + v (vTw) w^T = I - v (v^T - vTw w^T) - w w^T
            Chunk<T> vTw{};
            for(int i = firstRow; i < nChunks; i++)
              fmadd(v[i], w[i], vTw);
            bcast_sum(vTw);

            int j = col+1;
            for(; j+1 < m; j+=2)
              applyRotation2x2(nChunks, firstRow, j, w, v, vTw, pdata, lda, pdataResult);

            for(; j < m; j++)
              applyRotation2(nChunks, firstRow, j, w, v, vTw, pdata, lda, pdataResult);
          }
          else if( col+1 < m )
          {
            applyRotation(nChunks, firstRow, col+1, v, pdata, lda, pdataResult);
          }

          pdata = pdataResult;
          lda = nChunks;
          std::swap(v,w);
        }
      }


      //! Helper function for combining multiple (upper triangular) matrices
      //!
      //! Appends the new block to a work matrix. Reduces the work matrix to upper triangular form when it reaches its maximal size
      //! (the size of the reserved memory for the work matrix).
      //!
      //! This is part of the TSQR algorithm where multiple blocks are first transformed to upper triangular form and then the resulting
      //! upper triangular matrices can be combined again and reduced to upper triangular form...
      //!
      //! @tparam T           underlying data type
      //!
      //! @param nSrc         number of chunks of rows of the new block
      //! @param m            number of columns
      //! @param pdataSrc     new block with dimension nSrc*m, does not need to be upper triangular
      //! @param ldaSrc       offset of columns in pdataSrc
      //! @param nChunks      number of rows of the work matrix
      //! @param pdataWork    work matrix with dimension nChunks*m (currently unused parts are zero)
      //! @param workOffset   row offset of the next zero block in the work matrix, adjusted on output
      //!
      template<typename T>
      void copyBlockAndTransformMaybe(int nSrc, int m, const Chunk<T>* pdataSrc, long long ldaSrc, int nChunks, Chunk<T>* pdataWork, int& workOffset)
      {
        const int mChunks = (m-1) / Chunk<T>::size + 1;

        // if there is not enough space, reduce to upper triangular form first
        if( workOffset + nSrc > nChunks )
        {
          transformBlock(nChunks, m, pdataWork, nChunks, pdataWork);
          workOffset = mChunks;
        }

        assert(workOffset + nSrc <= nChunks);

        // copy into work buffer
        for(int j = 0; j < m; j++)
          for(int i = 0; i < nSrc; i++)
            pdataWork[workOffset + i + nChunks*j] = pdataSrc[i + ldaSrc*j];
        workOffset += nSrc;
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

        const auto nChunks = 2*mChunks;

        // get required buffer
        std::unique_ptr<Chunk<T>[]> buff{new Chunk<T>[nChunks*m]};

        // copy to buffer
        for(int j = 0; j < m; j++)
          for(int i = 0; i < mChunks; i++)
            unaligned_load(invec+(i+j*mChunks)*Chunk<T>::size, buff[i+j*nChunks]);
        for(int j = 0; j < m; j++)
          for(int i = 0; i < mChunks; i++)
            unaligned_load(inoutvec+(i+j*mChunks)*Chunk<T>::size, buff[mChunks+i+j*nChunks]);

        transformBlock(nChunks, m, &buff[0], nChunks, &buff[0]);

        // copy back to inoutvec
        for(int j = 0; j < m; j++)
          for(int i = 0; i < mChunks; i++)
            unaligned_store(buff[i+j*nChunks], inoutvec+(i+j*mChunks)*Chunk<T>::size);
      }

      //! wrapper function for MPI (otherwise the function signature does not match!)
      template<typename T>
      void combineTwoBlocks_mpiOp(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype)
      {
        combineTwoBlocks<T>((const T*)invec, (T*)inoutvec, len, datatype);
      }
    }
  }


  template<typename T>
  void block_TSQR(const MultiVector<T>& M, Tensor2<T>& R, int reductionFactor = 4, bool mpiGlobal = true)
  {
    // calculate dimensions and block sizes
    const long long n = M.rows();
    const int m = M.cols();
    const int mChunks = (m-1) / Chunk<T>::size + 1;
    const int nChunks = (reductionFactor+1) * mChunks;
    const long long nTotalChunks = M.rowChunks();
    const long long nIter = nTotalChunks / nChunks;
    const long long lda = M.colStrideChunks();

    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"rows", "cols", "reductionFactor"},{n, m, reductionFactor}}, // arguments
        {{(1.+1./reductionFactor)*(n*m*(1.+m))*kernel_info::FMA<T>()}, // flops - extremely roughly estimated
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

    std::unique_ptr<Chunk<T>[]> psharedBuff(new Chunk<T>[mChunks*nMaxThreads*m]);
    std::unique_ptr<Chunk<T>[]> presultBuff(new Chunk<T>[mChunks*m]);

#pragma omp parallel
    {
      const auto& [iThread,nThreads] = internal::parallel::ompThreadInfo();

      std::unique_ptr<Chunk<T>[]> pdataSmall{new Chunk<T>[nChunks*m]};
      std::unique_ptr<Chunk<T>[]> plocalBuff{new Chunk<T>[nChunks*m]};

      // fill with zero
      for(int i = 0; i < nChunks; i++)
        for(int j = 0; j < m; j++)
          plocalBuff[i+nChunks*j] = Chunk<T>{};

      // index to the next free block in plocalBuff
      int localBuffOffset = 0;

#pragma omp for schedule(static)
      for(long long iter = 0; iter < nIter; iter++)
      {
        internal::HouseholderQR::transformBlock(nChunks, m, &M.chunk(nChunks*iter,0), lda, &pdataSmall[0]);

        internal::HouseholderQR::copyBlockAndTransformMaybe(mChunks, m, &pdataSmall[0], nChunks, nChunks, &plocalBuff[0], localBuffOffset);
      }
      // remainder (missing bottom part that is smaller than nChunk*Chunk::size rows
      if( iThread == nThreads-1 && nIter*nChunks < nTotalChunks )
      {
        const int nLastChunks = nTotalChunks-nIter*nChunks;
        internal::HouseholderQR::transformBlock(nLastChunks, m, &M.chunk(nIter*nChunks,0), lda, &pdataSmall[0]);
        internal::HouseholderQR::copyBlockAndTransformMaybe(std::min(nLastChunks,mChunks), m, &pdataSmall[0], nLastChunks, nChunks, &plocalBuff[0], localBuffOffset);
      }

      // check if we need an additional reduction of plocalBuff
      if( localBuffOffset > mChunks )
        internal::HouseholderQR::transformBlock(nChunks, m, &plocalBuff[0], nChunks, &plocalBuff[0]);

      int offset = iThread*mChunks;
      for(int j = 0; j < m; j++)
        for(int i = 0; i < mChunks; i++)
          psharedBuff[offset + i + nThreads*mChunks*j] = plocalBuff[i + nChunks*j];

#pragma omp barrier

#pragma omp master
      {
        if( nThreads > 1 )
        {
          // reduce shared buffer
          internal::HouseholderQR::transformBlock(nThreads*mChunks, m, &psharedBuff[0], nThreads*mChunks, &psharedBuff[0]);

          // compress result
          for(int j = 0; j < m; j++)
            for(int i = 0; i < mChunks; i++)
              presultBuff[i+mChunks*j] = psharedBuff[i+nThreads*mChunks*j];
        }
        else
        {
          // psharedBuff is already as small as possible
          std::swap(psharedBuff,presultBuff);
        }
      }
    } // omp parallel

    if( mpiGlobal )
    {
      const auto& [iProc,nProcs] = internal::parallel::mpiProcInfo();
      if( nProcs > 1 )
      {
        // register MPI reduction operation
        MPI_Op tsqrOp;
        if( MPI_Op_create(&internal::HouseholderQR::combineTwoBlocks_mpiOp<T>, 0, &tsqrOp) != MPI_SUCCESS )
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
