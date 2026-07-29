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
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_multivector_triangular_solve.hpp"
#include "pitts_multivector_triangular_mult.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_eigen.hpp"
#include "pitts_performance.hpp"
#include <iostream>
#include <stdexcept>
#include <charconv>



int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 6 )
    throw std::invalid_argument("Requires 5 arguments (n m k nIterTSQR nIterQRCP)!");

  long long n = 0, m = 0, k = 0, nIterTSQR = 0, nIterQRCP = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], k);
  std::from_chars(argv[4], argv[5], nIterTSQR);
  std::from_chars(argv[5], argv[6], nIterQRCP);

  using Type = double;
  PITTS::MultiVector<Type> X(n, m);
  PITTS::MultiVector<Type> Qright(k, m);

  PITTS::MultiVector<Type> Qleft(n, m);
  PITTS::MultiVector<Type> Y(k, m);

  randomize(X);
  randomize(Qright);

  PITTS::Tensor2<Type> R(m, m);
  PITTS::Tensor2<Type> C(m, m);

  PITTS::Tensor2<Type> tmpR(m, m);
  PITTS::MultiVector<Type> tmpX(n, m);
  Eigen::VectorX<long long> jpvt(m);
  std::vector<int> colsPermutation(m);
  PITTS::Tensor2<Type> tau(m,1);

  PITTS::MultiVector<Type> Qleft_ref(n, m);
  PITTS::MultiVector<Type> Y_ref(k, m);

  using EigenMatrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::ColPivHouseholderQR<EigenMatrix> qrcp;

  EigenMatrix X_copy(n, m);
  X_copy = ConstEigenMap(X);


  //---------- TSQR variant ----------
  const auto& run_TSQR = [&]()
  {
    // needs
    // copy(X, Qleft);
    // copy(QRight, Y);

    block_TSQR(Qleft, R, 0, false);

    const auto mapR = ConstEigenMap(R);

    qrcp.compute(mapR);
    const auto r = qrcp.rank();
    colsPermutation.resize(r);
    Eigen::Map<Eigen::VectorXi>(colsPermutation.data(), r) = qrcp.colsPermutation().indices().head(r);
    tmpR.resize(r, r);
    EigenMap(tmpR) = qrcp.matrixR().topLeftCorner(r,r).template triangularView<Eigen::Upper>();
    
    triangularSolve(Qleft, tmpR, colsPermutation);
    triangularMult(Y, tmpR, colsPermutation);
  };


  EigenMap(tmpR).setZero();
  //---------- QRCP variant ----------
  const auto& run_QRCP = [&]()
  {
#ifndef PITTS_DIRECT_MKL_GEMM

    qrcp.compute(X_copy);
    // X P = Q R   =>  X = Q R P^(-1)

    EigenMap(Qleft).noalias() = qrcp.householderQ() * EigenMatrix::Identity(n, m);

    // Y^T = R P^(-1) Qright^T  =>  Y = Qright P R^T
    EigenMatrix Rt = qrcp.matrixR().topRows(m).template triangularView<Eigen::Upper>().transpose();
    EigenMap(C).noalias() = qrcp.colsPermutation() * Rt;
    EigenMap(Y).noalias() = ConstEigenMap(Qright) * ConstEigenMap(C);
    //EigenMap(Y).noalias() = ConstEigenMap(Qright) * qrcp.colsPermutation() * qrcp.matrixR().topRows(m).template triangularView<Eigen::Upper>().transpose();

#else

    copy(X, tmpX);
    const auto ldx = tmpX.colStrideChunks()*PITTS::Chunk<Type>::size;
    jpvt.setZero();
    EigenMap(tau).setZero();
    LAPACKE_dgeqp3_64(LAPACK_COL_MAJOR, tmpX.rows(), tmpX.cols(), &tmpX(0,0), ldx, jpvt.data(), &tau(0,0));

    // copy R out of tmpX
    EigenMap(tmpR).setZero();
    LAPACKE_dlacpy_64(LAPACK_COL_MAJOR, 'U', m, m, &tmpX(0,0), ldx, &tmpR(0,0), tmpR.r1());

    // generate Q
    LAPACKE_dorgqr_64(LAPACK_COL_MAJOR, n, m, m, &tmpX(0,0), ldx, &tau(0,0));

    for(int i = 0; i < m; i++)
      jpvt(i) -= 1;
    EigenMap(C).noalias() = jpvt.asPermutation() * ConstEigenMap(tmpR).transpose();
    EigenMap(Y).noalias() = ConstEigenMap(Qright) * ConstEigenMap(C);

#endif
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


  //---------- run QRCP variant ----------
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
    //EigenMap(Qleft).noalias() = ConstEigenMap(tmpX).leftCols(m);
    //std::cout << " difference in Qleft*Y^T: " << (ConstEigenMap(Qleft) * ConstEigenMap(Y).transpose() - ConstEigenMap(Qleft_ref)*ConstEigenMap(Y_ref).transpose()).array().abs().maxCoeff() << "\n";
  }



  PITTS::finalize(true);

  return 0;
}
