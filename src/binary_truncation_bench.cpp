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


/*
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
*/


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 8 )
    throw std::invalid_argument("Requires 7 arguments (n m k m_ nIterTSQR nIterQRCP nIterSVD)!");

  long long n = 0, m = 0, k = 0, m_ = 0, nIterTSQR = 0, nIterQRCP = 0, nIterSVD = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], k);
  std::from_chars(argv[4], argv[5], m_);
  std::from_chars(argv[5], argv[6], nIterTSQR);
  std::from_chars(argv[6], argv[7], nIterQRCP);
  std::from_chars(argv[7], argv[8], nIterSVD);

  using Type = double;
  PITTS::MultiVector<Type> X(n, m);
  PITTS::MultiVector<Type> Qright(k, m);

  PITTS::MultiVector<Type> Qleft(n, m_);
  PITTS::MultiVector<Type> Y(k, m_);

  randomize(X);
  randomize(Qright);

  PITTS::Tensor2<Type> R(m, m);
  PITTS::Tensor2<Type> C(m, m_);

  PITTS::Tensor2<Type> tmpR(m_, m);
  PITTS::MultiVector<Type> tmpX(n, m);
  Eigen::VectorX<long long> jpvt(m);
  PITTS::Tensor2<Type> tau(m,1);

  //PITTS::MultiVector<Type> Qleft_ref(n, m_);
  //PITTS::MultiVector<Type> Y_ref(k, m_);

  using EigenMatrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::ColPivHouseholderQR<EigenMatrix> qrcp;
#if EIGEN_VERSION_AT_LEAST(3,4,90)
  Eigen::BDCSVD<EigenMatrix, Eigen::ComputeThinV | Eigen::ComputeThinU> svd;
#else
  Eigen::BDCSVD<EigenMatrix> svd;
#endif

  EigenMatrix X_copy(n, m);
  X_copy = ConstEigenMap(X);


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
    auto sigma = svd.singularValues().topRows(m_);
    EigenMap(C) = V * sigma.array().inverse().matrix().asDiagonal();

    transform(X, C, Qleft);

    EigenMap(C) = V * sigma.asDiagonal();
    transform(Qright, C, Y);
  };


  EigenMap(tmpR).setZero();
  //---------- QRCP variant ----------
  const auto& run_QRCP = [&]()
  {
#ifndef PITTS_DIRECT_MKL_GEMM

    qrcp.compute(X_copy);
    // X P = Q R   =>  X = Q R P^(-1)

    EigenMap(Qleft).noalias() = qrcp.householderQ() * EigenMatrix::Identity(n, m_);

    // Y^T = R P^(-1) Qright^T  =>  Y = Qright P R^T
    EigenMatrix Rt = qrcp.matrixR().topRows(m_).template triangularView<Eigen::Upper>().transpose();
    EigenMap(C).noalias() = qrcp.colsPermutation() * Rt;
    EigenMap(Y).noalias() = ConstEigenMap(Qright) * ConstEigenMap(C);
    //EigenMap(Y).noalias() = ConstEigenMap(Qright) * qrcp.colsPermutation() * qrcp.matrixR().topRows(m_).template triangularView<Eigen::Upper>().transpose();

#else

    copy(X, tmpX);
    const auto ldx = tmpX.colStrideChunks()*PITTS::Chunk<Type>::size;
    jpvt.setZero();
    EigenMap(tau).setZero();
    LAPACKE_dgeqp3_64(LAPACK_COL_MAJOR, tmpX.rows(), tmpX.cols(), &tmpX(0,0), ldx, jpvt.data(), &tau(0,0));

    // copy R out of tmpX
    EigenMap(tmpR).setZero();
    LAPACKE_dlacpy_64(LAPACK_COL_MAJOR, 'U', m_, m, &tmpX(0,0), ldx, &tmpR(0,0), tmpR.r1());

    // generate Q
    LAPACKE_dorgqr_64(LAPACK_COL_MAJOR, n, m_, m_, &tmpX(0,0), ldx, &tau(0,0));

    for(int i = 0; i < m; i++)
      jpvt(i) -= 1;
    EigenMap(C).noalias() = jpvt.asPermutation() * ConstEigenMap(tmpR).transpose();
    EigenMap(Y).noalias() = ConstEigenMap(Qright) * ConstEigenMap(C);

#endif
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

    auto U = svd.matrixU().leftCols(m_);
    auto sigma = svd.singularValues().topRows(m_);
    auto V = svd.matrixV().leftCols(m_);

    EigenMap(Qleft).noalias() = U;

    EigenMap(C).noalias() = V * sigma.asDiagonal();
    EigenMap(Y).noalias() = ConstEigenMap(Qright) * ConstEigenMap(C);
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

    //copy(Qleft, Qleft_ref);
    //copy(Y, Y_ref);
  }


  //---------- run SVD variant ----------
  if( nIterQRCP > 0 )
  {
    run_QRCP();
    run_QRCP();

    double wtime = omp_get_wtime();
    for(int iter = 0; iter < nIterQRCP; iter++)
      run_QRCP();
    wtime = (omp_get_wtime() - wtime) / nIterQRCP;
    std::cout << "wtime QRCP variant: " << wtime << std::endl;


    // sign in B, Q is not unique...
    //EigenMap(Qleft).noalias() = ConstEigenMap(tmpX).leftCols(m_);
    //std::cout << " difference in Qleft*Y^T: " << (ConstEigenMap(Qleft) * ConstEigenMap(Y).transpose() - ConstEigenMap(Qleft_ref)*ConstEigenMap(Y_ref).transpose()).array().abs().maxCoeff() << "\n";
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
    //std::cout << " difference in Qleft*Y^T: " << (ConstEigenMap(Qleft) * ConstEigenMap(Y).transpose() - ConstEigenMap(Qleft_ref)*ConstEigenMap(Y_ref).transpose()).array().abs().maxCoeff() << "\n";
  }

  PITTS::finalize(true);

  return 0;
}
