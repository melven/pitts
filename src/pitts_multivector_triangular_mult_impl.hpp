// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_triangular_mult_impl.hpp
* @brief in-place matrix multiplication of a tall-skinny matrix and a small upper triangular matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-07-29
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TRIANGULAR_MULT_IMPL_HPP
#define PITTS_MULTIVECTOR_TRIANGULAR_MULT_IMPL_HPP

// includes
#include <omp.h>
#include <stdexcept>
#include <memory>
#include "pitts_multivector_triangular_mult.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_machine_info.hpp"
#include "pitts_eigen.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#ifdef PITTS_DIRECT_MKL_GEMM
#include <mkl_cblas.h>
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
#ifdef PITTS_DIRECT_MKL_GEMM
    inline void cblas_trmm_mapper(const CBLAS_LAYOUT Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE transa, const CBLAS_DIAG diag, const CBLAS_INDEX M, const CBLAS_INDEX N, const double alpha, const double * A, const CBLAS_INDEX lda, double * B, const CBLAS_INDEX ldb)
    {
      cblas_dtrmm(Layout, side, uplo, transa, diag, M, N, alpha, A, lda, B, ldb);
    }

    inline void cblas_trmm_mapper(const CBLAS_LAYOUT Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE transa, const CBLAS_DIAG diag, const CBLAS_INDEX M, const CBLAS_INDEX N, const float alpha, const float * A, const CBLAS_INDEX lda, float * B, const CBLAS_INDEX ldb)
    {
      cblas_strmm(Layout, side, uplo, transa, diag, M, N, alpha, A, lda, B, ldb);
    }
#endif
  }

  // implement multivector triangular multiplication
  template<typename T>
  void triangularMult(MultiVector<T>& X, const Tensor2<T>& R, const std::vector<int>& colsPermutation)
  {
    // special case: zero rank => dimensions are zero
    if( R.r1() == 0 && R.r2() == 0 && colsPermutation.empty() )
    {
      X.resize(X.rows(), 0);
      return;
    }

    if( colsPermutation.empty() )
    {
      // no column permutation: matrix dimensions must match
      if( R.r1() != R.r2() || R.r1() != X.cols() )
        throw std::invalid_argument("MultiVector::triangularMult: dimension mismatch!");
    }
    else
    {
      // with column permutation: only use subset of columns of X
      if( R.r1() != R.r2() || R.r1() != colsPermutation.size() || R.r1() > X.cols() )
        throw std::invalid_argument("MultiVector::triangularMult: dimension mismatch!");
      
      // check that the indices in colsPermutation are valid
      for(const int idx: colsPermutation)
      {
        if( idx >= X.cols() )
          throw std::invalid_argument("MultiVector::triangularMult: invalid column permutation index!");
      }
    }

    const MachineInfo mi = getMachineInfo();
    bool use_streaming_stores = X.rows()*R.r2()*sizeof(T) > 3*mi.cacheSize_L3_total;

    // perform streaming store for column i if it is not read
    const std::vector<bool> streamingStore = [&]()
    {
      if( colsPermutation.empty() )
        return std::vector<bool>(X.cols(), false);
      
      std::vector<bool> tmp(X.cols(), use_streaming_stores);
      for(const int idx: colsPermutation)
        tmp[idx] = false;
      return tmp;
    }();
    // amount of data updated (if cols > r1)
    const int streamingCols = [&]()
    {
      int tmp = 0;
      for(int i = 0; i < R.r2(); i++)
        tmp += (int)streamingStore[i];
      return tmp;
    }();

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"Xrows", "Xcols", "r", "streamingCols"},{X.rows(),X.cols(),R.r1(),streamingCols}}, // arguments
        {{(0.5*X.rows()*R.r1()*R.r2())*kernel_info::FMA<T>()}, // flops
         {(0.5*R.r1()*R.r2())*kernel_info::Load<T>() +
          (double(X.rows())*(R.r2()-streamingCols))*kernel_info::Update<T>() +
          (double(X.rows())*streamingCols)*(kernel_info::Store<T>()+kernel_info::Load<T>())}} // data transfers
        );


    // dimensions
    const long long nTotalChunks = X.rowChunks();
    const long long ldX = X.colStrideChunks();
    const int m = R.r2();
    // for small m, problem is memory-bound => optimize for simple code, small overhead, n.t.-store where applicable
    // for large m, problem is compute-bound, but only if we handle enough rows at once!
    constexpr int nChunks = 10;
    //int nMaxThreads = omp_get_max_threads();
    //nChunks = std::min<long long>(nChunks, nTotalChunks / (2*nMaxThreads));
    //nChunks = std::max(1, nChunks);
    
    constexpr int rowBlockSize = 2;
    constexpr int colBlockSize = 60;

    // optimization for smaller cases
    if( m <= colBlockSize && ldX > 1 && colsPermutation.empty() )
    {
      {
#pragma omp parallel for schedule(static)
        for(long long iChunk = 0; iChunk < nTotalChunks; iChunk+=2)
        {
          // offset in input vector
          Chunk<T>* pX = &X.chunk(iChunk, 0);

          if( iChunk+1 < ldX )
          {
            int j = 0;
            for(; j+1 < m; j+=2)
            {
              Chunk<T> tmp00{};
              Chunk<T> tmp01{};
              Chunk<T> tmp10{};
              Chunk<T> tmp11{};
              fmadd(R(j+0,j), pX[j*ldX+0], tmp00);
              fmadd(R(j+0,j), pX[j*ldX+1], tmp01);
              for(int k = j+1; k < m; k++)
              {
                  fmadd(R(j+0,k), pX[k*ldX+0], tmp00);
                  fmadd(R(j+0,k), pX[k*ldX+1], tmp01);
                  fmadd(R(j+1,k), pX[k*ldX+0], tmp10);
                  fmadd(R(j+1,k), pX[k*ldX+1], tmp11);
              }
              pX[(j+0)*ldX+0] = tmp00;
              pX[(j+0)*ldX+1] = tmp01;
              pX[(j+1)*ldX+0] = tmp10;
              pX[(j+1)*ldX+1] = tmp11;
            }
            for(; j < m; j++)
            {
              Chunk<T> tmp0{};
              Chunk<T> tmp1{};
              for(int k = j; k < m; k++)
              {
                fmadd(R(j,k), pX[k*ldX+0], tmp0);
                fmadd(R(j,k), pX[k*ldX+1], tmp1);
              }
              pX[(j+0)*ldX+0] = tmp0;
              pX[(j+0)*ldX+1] = tmp1;
            }
          }
          else // iChunk+1 == ldX
          {
            for(int j = 0; j < m; j++)
            {
              Chunk<T> tmp0{};
              for(int k = j; k < m; k++)
                fmadd(R(j,k), pX[k*ldX+0], tmp0);
              pX[(j+0)*ldX+0] = tmp0;
            }
          }
        }
      }

      // reshape to resulting size
      X.resize(X.rows(), m);

      return;
    }

#pragma omp parallel
    {
      std::unique_ptr<Chunk<T>[]> buff(new Chunk<T>[m*nChunks]);

#pragma omp for schedule(static)
      for(long long iChunk = 0; iChunk < nTotalChunks; iChunk+=nChunks)
      {
        const int nRemainingChunks = std::min<long long>(nChunks, nTotalChunks-iChunk);

        // copy data to buff
        for(int j = 0; j < m; j++)
        {
          int piv_j = j;
          if( !colsPermutation.empty() )
            piv_j = colsPermutation[j];
          for(int i = 0; i < nRemainingChunks; i++)
            buff[j*nChunks+i] = X.chunk(iChunk+i, piv_j);
          for(int i = nRemainingChunks; i < nChunks; i++)
            buff[j*nChunks+i] = Chunk<T>{};
        }


        using EigenMatrix = Eigen::MatrixX<T>;
        auto mapBuff = Eigen::Map<EigenMatrix, EigenAligned>(&buff[0][0], nChunks*Chunk<T>::size, m);
        auto mapR = ConstEigenMap(R);

#ifndef PITTS_DIRECT_MKL_GEMM
        mapBuff = mapBuff * mapR.template triangularView<Eigen::Upper>().transpose();
#else
        internal::cblas_trmm_mapper(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, mapBuff.rows(), mapBuff.cols(), T(1), mapR.data(), mapR.colStride(), mapBuff.data(), mapBuff.colStride());
#endif


        // store result
        for(int j = 0; j < m; j++)
          for(int i = 0; i < nRemainingChunks; i++)
          {
            if( streamingStore[j] )
              streaming_store(buff[j*nChunks+i], X.chunk(iChunk+i, j));
            else
              X.chunk(iChunk+i,j) = buff[j*nChunks+i];
          }
      }
    }
    
    // reshape to resulting size
    X.resize(X.rows(), m);
  }
}


#endif // PITTS_MULTIVECTOR_TRIANGULAR_MULT_IMPL_HPP
