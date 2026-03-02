// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_machine_info.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_performance.hpp"
#include "pitts_common.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensor3_random.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include <iostream>
#include <charconv>
#include <stdexcept>


namespace PITTS
{
namespace internal
{
#ifdef PITTS_DIRECT_MKL_GEMM
    inline void cblas_gemm_mapper4(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const double alpha, const double * A, const CBLAS_INDEX lda, const double * B, const CBLAS_INDEX ldb, const double beta, double * C, const CBLAS_INDEX ldc)
    {
      cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    inline void cblas_gemm_mapper4(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const float alpha, const float * A, const CBLAS_INDEX lda, const float * B, const CBLAS_INDEX ldb, const float beta, float * C, const CBLAS_INDEX ldc)
    {
      cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
#endif
}

template<typename T>
void ttn_local_gradient_contract(const Tensor3<T>& optTensor, const MultiVector<T>& envLeft, const MultiVector<T>& envRight,  const MultiVector<T>& envTop, Tensor3<T>& gradOptTensor, bool mpiParallel)
{
  if( envLeft.rows() != envRight.rows() || envLeft.rows() != envTop.rows() )
    throw std::invalid_argument("PITTS::ttn_local_gradient_contract: envLeft, envRight and envTop need the same number of rows!");
  if( envLeft.cols() != optTensor.r1() || envRight.cols() != optTensor.n() || envTop.cols() % optTensor.r2() != 0 )
    throw std::invalid_argument("PITTS::ttn_local_gradient_contract: envLeft/Right/Top tensor dimensions do not match optTensor dimensions!");

  const auto nSampleChunks = envLeft.rowChunks();
  const auto b_left = optTensor.r1();
  const auto b_right = optTensor.n();
  const auto b_top = optTensor.r2();
  const auto n_classes = envTop.cols() / b_top;
  const auto n_samples = nSampleChunks * Chunk<T>::size;

  const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
      {{"b_left", "b_right", "b_top", "n_classes", "n_samples"},{b_left, b_right, b_top, n_classes, n_samples}}, // arguments
      {{(b_left*b_right*n_samples+n_samples*b_right*b_top)*kernel_info::Mult<T>() + (2*n_samples*b_left*b_right*b_top+2*n_samples*b_top*n_classes)*kernel_info::FMA<T>()}, // flops - roughly estimated
       {(b_left*b_right*b_top+n_samples*(b_left+b_right+n_classes*b_top))*kernel_info::Load<T>() + (b_left+b_right+b_top)*kernel_info::Store<T>()}} // data transfers
      );


  // calculate performance tuning parameters
  int nChunks;
  {
    const MachineInfo mi = getMachineInfo();
    // L2 cache size per core (in chunks)
    int cacheSize_L2 = (mi.cacheSize_L2_perCore > 0 ? mi.cacheSize_L2_perCore : 1*1024*1024) / (Chunk<T>::size * sizeof(T));

    nChunks = std::max<int>(2, 0.75*cacheSize_L2 / (b_left + b_right + b_left*b_right + b_top + n_classes + b_top*n_classes));
    std::cout << "nChunks: " << nChunks << "\n";
  }


  gradOptTensor.resize(b_left, b_right, b_top);
  gradOptTensor.setConstant(0);
  auto size = gradOptTensor.r1() * gradOptTensor.n() * gradOptTensor.r2();
  T* gradOptData = &gradOptTensor.data()[0][0];
#pragma omp parallel reduction(+:gradOptData[:size])
  {
    /*
       ============================== contraction path  ==============================
       Complete contraction:  ijb,in,jn,lbn,xn,yn,lcn->cxy
       Naive scaling:  8
       Optimized scaling:  4
       Naive FLOP count:  1.094e+18
       Optimized FLOP count:  5.250e+10
       Theoretical speedup:  2.083e+7
       Largest intermediate:  2.500e+8 elements
       --------------------------------------------------------------------------------
       scaling        BLAS                current                             remaining
       --------------------------------------------------------------------------------
       3              0             jn,in->jni            ijb,lbn,xn,yn,lcn,jni->cxy
       4           TDOT            jni,ijb->nb                 lbn,xn,yn,lcn,nb->cxy
       3              0             nb,lbn->nl                     xn,yn,lcn,nl->cxy
       3              0             nl,lcn->nc                         xn,yn,nc->cxy
       3              0             nc,xn->ncx                           yn,ncx->cxy
       4           GEMM            ncx,yn->cxy                              cxy->cxy
       */

    //oe.contract('ijb,ni,nj,nlb,nx,ny,nlc->cxy', optTensor, envLeft, envRight, envTop, envLeft, envRight, envTop, optimize='optimal', out=gradOptTensor)

    auto [iThread,nThreads] = internal::parallel::ompThreadInfo();
    const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(nSampleChunks, {iThread, nThreads});
    int nLocalChunks = std::min<int>(nChunks, nLast - nFirst + 1);
    int nLocal = nLocalChunks*Chunk<T>::size;

    MultiVector<T> buff_1(nLocal, std::max(std::max(b_left*b_right, n_classes), b_top*b_right)); // jin, nl, ncy
    MultiVector<T> buff_2(nLocal, b_top); // nb, nc


    // iterate over chunks
    for(int iChunkOffset = nFirst; iChunkOffset <= nLast; iChunkOffset += nLocalChunks)
    {
      // truncate for last iter
      int nCurrChunks = std::min<int>(nLocalChunks, nLast-iChunkOffset+1);

      // ni,nj->nij
      buff_1.resize(nLocal, b_left*b_right);
      for(int j = 0; j < b_right; j++)
        for(int i = 0; i < b_left; i++)
          for(int iChunk = 0; iChunk < nCurrChunks; iChunk++)
            mul(envLeft.chunk(iChunk+iChunkOffset, i), envRight.chunk(iChunk+iChunkOffset, j), buff_1.chunk(iChunk, i+j*b_left));

      // nij,ijb->nb
      {
        auto mapA = ConstEigenMap(buff_1);
        auto mapB = ConstEigenMap(unfold_left(optTensor));
        auto mapC = EigenMap(buff_2);
#ifndef PITTS_DIRECT_MKL_GEMM
        mapC.noalias() = mapA * mapB;
#else
        internal::cblas_gemm_mapper4(CblasColMajor, CblasNoTrans, CblasNoTrans, mapC.rows(), mapC.cols(), mapA.cols(), T(1), mapA.data(), mapA.colStride(), mapB.data(), mapB.colStride(), T(0), mapC.data(), mapC.colStride());
#endif
      }

      // nb,lbn->nl
      buff_1.resize(nLocal, n_classes);
      for(int iChunk = 0; iChunk < nLocalChunks; iChunk++)
      {
        for(int i = 0; i < n_classes; i++)
          buff_1.chunk(iChunk, i) = Chunk<T>{};
        for(int j = 0; j < b_top; j++)
          for(int i = 0; i < n_classes; i++)
            fmadd(buff_2.chunk(iChunk,j), envTop.chunk(iChunk+iChunkOffset,i+j*n_classes), buff_1.chunk(iChunk, i));
      }

      // nl,lcn->nc
      buff_2.resize(nLocal, b_top);
      for(int iChunk = 0; iChunk < nLocalChunks; iChunk++)
      {
        for(int j = 0; j < b_top; j++)
          buff_2.chunk(iChunk,j) = Chunk<T>{};
        for(int i = 0; i < n_classes; i++)
          for(int j = 0; j < b_top; j++)
            fmadd(buff_1.chunk(iChunk,i), envTop.chunk(iChunk+iChunkOffset,i+j*n_classes), buff_2.chunk(iChunk,j));
      }

      // nc,yn->nyc
      buff_1.resize(nLocal, b_right*b_top);
      for(int iChunk = 0; iChunk < nLocalChunks; iChunk++)
        for(int j = 0; j < b_top; j++)
          for(int i = 0; i < b_right; i++)
            mul(buff_2.chunk(iChunk,j), envRight.chunk(iChunk+iChunkOffset, i), buff_1.chunk(iChunk, i+j*b_right));

      // nx,nyc->xyc
      {
        // envLeft might have fewer data if chunks are not used fully (padded with zeros!)
        using mat = Eigen::MatrixX<T>;
        auto mapA = Eigen::Map<const mat, EigenAligned, Eigen::OuterStride<> >(&envLeft.chunk(iChunkOffset,0)[0], nLocal, b_left, Eigen::OuterStride<>(envLeft.colStrideChunks()*Chunk<T>::size));
        auto mapB = ConstEigenMap(buff_1);
        auto mapC = Eigen::Map<mat>(gradOptData, b_left, b_right*b_top);
#ifndef PITTS_DIRECT_MKL_GEMM
        mapC.noalias() += mapA.transpose() * mapB;
#else
        internal::cblas_gemm_mapper4(CblasColMajor, CblasTrans, CblasNoTrans, mapC.rows(), mapC.cols(), mapA.rows(), T(1), mapA.data(), mapA.colStride(), mapB.data(), mapB.colStride(), T(1), mapC.data(), mapC.colStride());
#endif
      }
    }
  }

  if( mpiParallel )
  {
    if( MPI_Allreduce(MPI_IN_PLACE, gradOptTensor.data(), size, internal::parallel::mpiType<T>(), MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS )
      throw std::runtime_error("Failure returned from MPI_Allreduce");
  }
}

}


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 7 )
    throw std::invalid_argument("Requires 6 arguments (b_left b_right b_top n_classes n_samples nIter)!");

  long long b_left = 0, b_right = 0, b_top = 0, n_classes = 0, n_samples = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], b_left);
  std::from_chars(argv[2], argv[3], b_right);
  std::from_chars(argv[3], argv[4], b_top);
  std::from_chars(argv[4], argv[5], n_classes);
  std::from_chars(argv[5], argv[6], n_samples);
  std::from_chars(argv[6], argv[7], nIter);

  bool mpiParallel = false;
  {
    const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();
    const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(n_samples, {iProc,nProcs});
    n_samples = nLast - nFirst + 1;
    if( nProcs > 0 )
      mpiParallel = true;
  }

  using Type = double;
  PITTS::Tensor3<Type> optTensor(b_left, b_right, b_top);
  PITTS::Tensor3<Type> gradOptTensor(b_left, b_right, b_top);
  PITTS::MultiVector<Type> envLeft(n_samples, b_left), envRight(n_samples, b_right), envTop(n_samples, b_top * n_classes);

  randomize(optTensor);
  randomize(gradOptTensor);
  randomize(envLeft);
  randomize(envRight);
  randomize(envTop);

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    ttn_local_gradient_contract(optTensor, envLeft, envRight, envTop, gradOptTensor, mpiParallel);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
