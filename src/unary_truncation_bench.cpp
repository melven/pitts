// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "omp.h"
#include "pitts_mkl.hpp"
#include "pitts_common.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_eigen.hpp"
#include "pitts_performance.hpp"
#include <iostream>
#include <stdexcept>
#include <charconv>


namespace
{
#ifdef PITTS_DIRECT_MKL_GEMM
  inline void cblas_gemm_mapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const double alpha, const double * A, const CBLAS_INDEX lda, const double * B, const CBLAS_INDEX ldb, const double beta, double * C, const CBLAS_INDEX ldc)
  {
    cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline void cblas_gemm_mapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const float alpha, const float * A, const CBLAS_INDEX lda, const float * B, const CBLAS_INDEX ldb, const float beta, float * C, const CBLAS_INDEX ldc)
  {
    cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline void cblas_gemm_mapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const std::complex<double> alpha, const std::complex<double> * A, const CBLAS_INDEX lda, const std::complex<double> * B, const CBLAS_INDEX ldb, const std::complex<double> beta, std::complex<double> * C, const CBLAS_INDEX ldc)
  {
    cblas_zgemm(layout, TransA, TransB, M, N, K, (const MKL_Complex16*)&alpha, (const MKL_Complex16*)A, lda, (const MKL_Complex16*)B, ldb, (const MKL_Complex16*)&beta, (MKL_Complex16*)C, ldc);
  }

  inline void cblas_gemm_mapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N, const CBLAS_INDEX K, const std::complex<float> alpha, const std::complex<float> * A, const CBLAS_INDEX lda, const std::complex<float> * B, const CBLAS_INDEX ldb, const std::complex<float> beta, std::complex<float> * C, const CBLAS_INDEX ldc)
  {
    cblas_cgemm(layout, TransA, TransB, M, N, K, (const MKL_Complex8*)&alpha, (const MKL_Complex8*)A, lda, (const MKL_Complex8*)B, ldb, (const MKL_Complex8*)&beta, (MKL_Complex8*)C, ldc);
  }
#endif
}


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 6 )
    throw std::invalid_argument("Requires 5 arguments (n m m_ nIterTSQR nIterSVD)!");

  long long n = 0, m = 0, m_ = 0, nIterTSQR = 0, nIterSVD = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], m_);
  std::from_chars(argv[4], argv[5], nIterTSQR);
  std::from_chars(argv[5], argv[6], nIterSVD);

  using Type = double;
  PITTS::MultiVector<Type> X(n, m);
  PITTS::MultiVector<Type> B(n, m_);
  //PITTS::MultiVector<Type> B_ref(n, m_);
  randomize(X);
  PITTS::Tensor2<Type> R(m, m);
  PITTS::Tensor2<Type> Q(m, m_);
  //PITTS::Tensor2<Type> Q_ref(m, m_);

  using EigenMatrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
#if EIGEN_VERSION_AT_LEAST(3,4,90)
  Eigen::BDCSVD<EigenMatrix, Eigen::ComputeThinV | Eigen::ComputeThinU> svd;
#else
  Eigen::BDCSVD<EigenMatrix> svd;
#endif


  //---------- TSQR variant ----------
  const auto& run_TSQR = [&]()
  {
    block_TSQR(X, R, 0, false);

    const auto mapR = ConstEigenMap(R);

#if EIGEN_VERSION_AT_LEAST(3,4,90)
    svd.compute(mapR);
#else
    svd.compute(mapR, Eigen::ComputeThinV | Eigen::ComputeThinU);
#endif

    // another copying around, hopefully not so critical in the overall time
    auto V = svd.matrixV().leftCols(m_);
    EigenMap(Q) = V;

    const auto mapX = ConstEigenMap(X);
    auto mapB = EigenMap(B);

    transform(X, Q, B);
//#ifndef PITTS_DIRECT_MKL_GEMM
//    mapB.noalias() = mapX * V;
//#else
//    cblas_gemm_mapper(CblasColMajor, CblasNoTrans, CblasNoTrans, mapB.rows(), mapB.cols(), mapX.cols(), Type(1), mapX.data(), mapX.colStride(), V.data(), V.colStride(), Type(0), mapB.data(), mapB.colStride());
//#endif
  };


  //---------- SVD variant ----------
  const auto& run_SVD = [&]()
  {
    const auto mapX = ConstEigenMap(X);
#if EIGEN_VERSION_AT_LEAST(3,4,90)
    svd.compute(mapX);
#else
    svd.compute(mapX, Eigen::ComputeThinV | Eigen::ComputeThinU);
#endif

    EigenMap(Q).noalias() = svd.matrixV().leftCols(m_);
    EigenMap(B).noalias() = svd.matrixU().leftCols(m_) * svd.singularValues().asDiagonal();
  };


  //---------- run TSQR variant ----------
  if( nIterTSQR > 0 )
  {
    run_TSQR();
    run_TSQR();

    PITTS::performance::clearStatistics();

    double wtime = omp_get_wtime();
    for(int iter = 0; iter < nIterTSQR; iter++)
      run_TSQR();
    wtime = (omp_get_wtime() - wtime) / nIterTSQR;
    std::cout << "wtime TSQR variant: " << wtime << std::endl;

    //copy(B, B_ref);
    //copy(Q, Q_ref);
  }

  //---------- run SVD variant ----------
  if( nIterSVD > 0 )
  {
    run_SVD();
    run_SVD();

    double wtime = omp_get_wtime();
    for(int iter = 0; iter < nIterSVD; iter++)
      run_SVD();
    wtime = (omp_get_wtime() - wtime) / nIterSVD;
    std::cout << "wtime SVD variant: " << wtime << std::endl;

    // sign in B, Q is not unique...
    //std::cout << " difference in BQ^T: " << (ConstEigenMap(B) * ConstEigenMap(Q).transpose() - ConstEigenMap(B_ref)*ConstEigenMap(Q_ref).transpose()).array().abs().maxCoeff() << "\n";
  }

  PITTS::finalize(true);

  return 0;
}
