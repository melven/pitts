// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_common.hpp"
#include "pitts_parallel.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensor3_random.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include <iostream>
#include <charconv>
#include <stdexcept>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace internal
{
#ifdef PITTS_DIRECT_MKL_GEMM
    inline void cblas_gemm_mapper5(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const double alpha, const double * A, const CBLAS_INDEX lda, const double * B, const CBLAS_INDEX ldb, const double beta, double * C, const CBLAS_INDEX ldc)
    {
      cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    inline void cblas_gemm_mapper5(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const float alpha, const float * A, const CBLAS_INDEX lda, const float * B, const CBLAS_INDEX ldb, const float beta, float * C, const CBLAS_INDEX ldc)
    {
      cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
#endif
}

template<typename T>
void ttn_sum_over_outer_product(const MultiVector<T>& envLeft, const MultiVector<T>& envRight,  const MultiVector<T>& envTop, Tensor3<T>& result, bool mpiParallel)
{
  if( envLeft.rows() != envRight.rows() || envLeft.rows() != envTop.rows() )
    throw std::invalid_argument("PITTS::ttn_local_gradient_contract: envLeft, envRight and envTop need the same number of rows!");

  result.resize(envLeft.cols(), envTop.cols(), envRight.cols());

  const auto nSampleChunks = envLeft.rowChunks();
  const auto b_left = envLeft.cols();
  const auto b_right = envRight.cols();
  const auto b_top = envTop.cols();
  const auto n_samples = nSampleChunks * Chunk<T>::size;

  const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
      {{"b_left", "b_right", "b_top", "n_samples"},{b_left, b_right, b_top, n_samples}}, // arguments
      {{(b_left*b_top*n_samples)*kernel_info::Mult<T>() + (n_samples*b_left*b_right*b_top)*kernel_info::FMA<T>()}, // flops - roughly estimated
       {n_samples*(b_left+b_right+b_top)*kernel_info::Load<T>() + (b_left+b_right+b_top)*kernel_info::Store<T>()}} // data transfers
      );


  // calculate performance tuning parameters
  int nChunks;
  {
    const MachineInfo mi = getMachineInfo();
    // L2 cache size per core (in chunks)
    int cacheSize_L2 = (mi.cacheSize_L2_perCore > 0 ? mi.cacheSize_L2_perCore : 1*1024*1024) / (Chunk<T>::size * sizeof(T));

    nChunks = std::max<int>(2, 0.75*cacheSize_L2 / (b_left + b_right + b_top + b_left*b_top));
    std::cout << "nChunks: " << nChunks << "\n";
  }


  result.setConstant(0);
  auto size = result.r1() * result.n() * result.r2();
  T* resultData = &result.data()[0][0];
#pragma omp parallel reduction(+:resultData[:size])
  {
    auto [iThread,nThreads] = internal::parallel::ompThreadInfo();
    const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(nSampleChunks, {iThread, nThreads});
    int nLocalChunks = std::min<int>(nChunks, nLast - nFirst + 1);
    int nLocal = nLocalChunks*Chunk<T>::size;

    MultiVector<T> buffer(nLocal, b_left*b_top);


    // iterate over chunks
    for(long long iChunkOffset = nFirst; iChunkOffset <= nLast; iChunkOffset += nLocalChunks)
    {
      // truncate for last iter
      int nCurrChunks = std::min<long long>(nLocalChunks, nLast-iChunkOffset+1);
      long long iOffset = iChunkOffset*Chunk<T>::size;
      int nCurr = std::min<long long>(nCurrChunks*Chunk<T>::size, envLeft.rows() - iOffset);

      for(int j = 0; j < b_top; j++)
        for(int i = 0; i < b_left; i++)
          for(int iChunk = 0; iChunk < nCurrChunks; iChunk++)
            mul(envLeft.chunk(iChunk+iChunkOffset, i), envTop.chunk(iChunk+iChunkOffset, j), buffer.chunk(iChunk, i+j*b_left));

      {
        auto mapA = ConstEigenMap(buffer).topRows(nCurr);
        auto mapB = ConstEigenMap(envRight).middleRows(iOffset, nCurr);
        using mat = Eigen::MatrixX<T>;
        auto mapC = Eigen::Map<mat>(resultData, b_left*b_top, b_right);
#ifndef PITTS_DIRECT_MKL_GEMM
        mapC.noalias() = mapA.transpose() * mapB;
#else
        internal::cblas_gemm_mapper5(CblasColMajor, CblasTrans, CblasNoTrans, mapC.rows(), mapC.cols(), nCurr, T(1), mapA.data(), mapA.colStride(), mapB.data(), mapB.colStride(), T(1), mapC.data(), mapC.colStride());
#endif
      }
    }
  }

  if( mpiParallel )
  {
    if( MPI_Allreduce(MPI_IN_PLACE, result.data(), size, internal::parallel::mpiType<T>(), MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS )
      throw std::runtime_error("Failure returned from MPI_Allreduce");
  }
}
}

int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 6 )
    throw std::invalid_argument("Requires 5 arguments (b_left b_right b_top n_samples nIter)!");

  long long b_left = 0, b_right = 0, b_top = 0, n_samples = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], b_left);
  std::from_chars(argv[2], argv[3], b_right);
  std::from_chars(argv[3], argv[4], b_top);
  std::from_chars(argv[4], argv[5], n_samples);
  std::from_chars(argv[5], argv[6], nIter);

  bool mpiParallel = false;
  {
    const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();
    const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(n_samples, {iProc,nProcs});
    n_samples = nLast - nFirst + 1;
    if( nProcs > 0 )
      mpiParallel = true;
  }

  using Type = double;
  PITTS::Tensor3<Type> result(b_left, b_top, b_right);
  PITTS::MultiVector<Type> envLeft(n_samples, b_left), envRight(n_samples, b_right), envTop(n_samples, b_top);

  randomize(result);
  randomize(envLeft);
  randomize(envRight);
  randomize(envTop);

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    ttn_sum_over_outer_product(envLeft, envRight,  envTop, result, mpiParallel);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
