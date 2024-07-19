// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_common.hpp"
#include "pitts_mkl.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_performance.hpp"
#include <iostream>
#include <charconv>
#include <stdexcept>
#include <cassert>

// pitts_tensor3_split_impl.hpp internal::normalize_svd_only

namespace
{
  using Type = double;
  using EigenMatrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using EigenQR = Eigen::ColPivHouseholderQR<EigenMatrix>;

  struct Work
  {
    EigenQR qr;
    PITTS::Tensor2<Type> tmpA, R, tau;
    Eigen::VectorX<int> jpvt;
  };


  //! small wrapper around QR, avoid reallocating svd data each time...
  void calculate_qr(const PITTS::Tensor2<Type>& M, Work& w)
  {
    using namespace PITTS;

    const auto n = M.r1();
    const auto m = M.r2();

    if( m > n )
      throw std::invalid_argument("Only supports n >= m!");

    w.tmpA.resize(n,m);

#ifdef PITTS_DIRECT_MKL_GEMM
    copy(M, w.tmpA);

    w.jpvt.resize(m);
    w.tau.resize(m,1);
    w.R.resize(m,m);

    w.jpvt.setZero();
    EigenMap(w.tau).setZero();
    EigenMap(w.R).setZero();
#endif

    using namespace PITTS;

    const auto timer = PITTS::performance::createScopedTimer<Tensor2<Type>>(
      {{"n", "m"},{n, m}}, // arguments
      {{(2.*n*m*m-2.*m*m*m/3)*kernel_info::FMA<Type>()}, // flops
       {(2.*n*m)*kernel_info::Update<Type>() + (2.*0.5*m*m)*kernel_info::Load<Type>() + (2.*0.5*m*m)*kernel_info::Store<Type>()}} // data transfers
      );

    const auto mapM = ConstEigenMap(M);

#ifndef PITTS_DIRECT_MKL_GEMM

    w.qr.compute(mapM);

    EigenMap(w.tmpA).noalias() = w.qr.householderQ() * EigenMatrix::Identity(n, m);

#else
    // unfortunately, there is no LAPACKE_dlatsqr yet, only the raw variant with work arrays in Fortran...
    LAPACKE_dgeqp3(LAPACK_COL_MAJOR, n, m, &w.tmpA(0,0), w.tmpA.r1(), w.jpvt.data(), &w.tau(0,0));

    //EigenMap(w.R).template triangularView<Eigen::Upper>().noalias() = ConstEigenMap(w.tmpA).template triangularView<Eigen::Upper>();
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', n, m, &w.tmpA(0,0), w.tmpA.r1(), &w.R(0,0), w.R.r1());

    LAPACKE_dorgqr(LAPACK_COL_MAJOR, n, m, m, &w.tmpA(0,0), w.tmpA.r1(), &w.tau(0,0));

    //for(int i = 0; i < m; i++)
    //  w.jpvt(i) -= 1;
    //std::cout << "Error: " << (ConstEigenMap(w.tmpA) * ConstEigenMap(w.R).template triangularView<Eigen::Upper>() - ConstEigenMap(M) * w.jpvt.asPermutation()).norm() << "\n";
#endif
  }

  using EigenUnpivotedQR = Eigen::HouseholderQR<EigenMatrix>;

  struct WorkUnpivoted
  {
    EigenUnpivotedQR qr;
    PITTS::Tensor2<Type> tmpA, Q, R, T, tau;
    int nb;
  };

  //! small wrapper around QR, avoid reallocating svd data each time...
  void calculate_unpivoted_qr(const PITTS::Tensor2<Type>& M, WorkUnpivoted& w)
  {
    using namespace PITTS;

    const auto n = M.r1();
    const auto m = M.r2();

    if( m > n )
      throw std::invalid_argument("Only supports n >= m!");

    w.tmpA.resize(n,m);

#ifdef PITTS_DIRECT_MKL_GEMM
    copy(M, w.tmpA);

    w.R.resize(m,m);
    w.Q.resize(n,m);
    w.tau.resize(m,1);

    EigenMap(w.R).setZero();

    const auto mi = PITTS::getMachineInfo();
    w.nb = 16 * std::max<int>(1, int(0.7 * mi.cacheSize_L2_perCore / sizeof(Type) / 16 / m));
    w.nb = std::min<int>(w.nb, m);
    w.T.resize(w.nb, m);
#endif

    using namespace PITTS;

    const auto timer = PITTS::performance::createScopedTimer<Tensor2<Type>>(
      {{"n", "m"},{n, m}}, // arguments
      {{(2.*n*m*m-2.*m*m*m/3)*kernel_info::FMA<Type>()}, // flops
       {(2.*n*m)*kernel_info::Update<Type>() + (2.*0.5*m*m)*kernel_info::Load<Type>() + (2.*0.5*m*m)*kernel_info::Store<Type>()}} // data transfers
      );

    const auto mapM = ConstEigenMap(M);

#ifndef PITTS_DIRECT_MKL_GEMM

    w.qr.compute(mapM);

    EigenMap(w.tmpA).noalias() = w.qr.householderQ() * EigenMatrix::Identity(n, m);

#else
    // try to use blocked variant
    //LAPACKE_dgeqrt(LAPACK_COL_MAJOR, n, m, w.nb, &w.tmpA(0,0), w.tmpA.r1(), &w.T(0,0), w.T.r1());
    //LAPACKE_dgeqr2(LAPACK_COL_MAJOR, n, m, &w.tmpA(0,0), w.tmpA.r1(), &w.tau(0,0));
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, m, &w.tmpA(0,0), w.tmpA.r1(), &w.tau(0,0));

    //EigenMap(w.R).template triangularView<Eigen::Upper>().noalias() = ConstEigenMap(w.tmpA).template triangularView<Eigen::Upper>();
    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', n, m, &w.tmpA(0,0), w.tmpA.r1(), &w.R(0,0), w.R.r1());

//#pragma omp parallel for schedule(static)
//    for(int i = 0; i < n; i++)
//      for(int j = 0; j < m; j++)
//        w.Q(i,j) = double(i == j);

    //LAPACKE_dgemqrt(LAPACK_COL_MAJOR, 'L', 'N', n, m, m, w.nb, &w.tmpA(0,0), w.tmpA.r1(), &w.T(0,0), w.T.r1(), &w.Q(0,0), w.Q.r1());
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, n, m, m, &w.tmpA(0,0), w.tmpA.r1(), &w.tau(0,0));

    //std::cout << "error: " << (ConstEigenMap(M) - ConstEigenMap(w.tmpA) * ConstEigenMap(w.R)).norm() << "\n";
    //std::cout << "error: " << (ConstEigenMap(M) - ConstEigenMap(w.Q) * ConstEigenMap(w.R)).norm() << "\n";
    //std::cout << "R:\n" << ConstEigenMap(w.R) << "\n";
    //std::cout << "Q^T Q:\n" << ConstEigenMap(w.Q).transpose() * ConstEigenMap(w.Q) << "\n";
#endif
  }


#ifdef PITTS_DIRECT_MKL_GEMM
  struct WorkUnpivotedGenericLAPACK
  {
    PITTS::Tensor2<Type> R;
    PITTS::MultiVector<Type> tmpA, Q;
    std::vector<double> work;
    std::vector<double> t;
  };

  //! small wrapper around QR, avoid reallocating svd data each time...
  void calculate_unpivoted_generic_lapack_qr(const PITTS::Tensor2<Type>& M, WorkUnpivotedGenericLAPACK& w)
  {
    using namespace PITTS;

    const int n = M.r1();
    const int m = M.r2();

    if( m > n )
      throw std::invalid_argument("Only supports n >= m!");

    w.tmpA.resize(n,m);
    int lda = w.tmpA.colStrideChunks()*PITTS::Chunk<Type>::size;

    //copy(M, w.tmpA);
#pragma omp parallel
    {
      for(int j = 0; j < m; j++)
      {
#pragma omp for schedule(static) nowait
        for(int i = 0; i < n; i++)
          w.tmpA(i,j) = M(i,j);
      }
    }

    w.R.resize(m,m);
    w.Q.resize(n,m);

    if( w.work.size() == 0 )
      EigenMap(w.R).setZero();
    //EigenMap(w.Q).setZero();

    // work-size query
    int tsize = -1;
    int lwork = -1;
    int info = 0;
    double optT = 0;
    double optWork = 0;
    dgeqr(&n, &m, &w.tmpA(0,0), &lda, &optT, &tsize, &optWork, &lwork, &info);
    //std::cout << "info: " << info << ", optT: " << optT << ", optWork: " << optWork << "\n";
    w.t.resize(optT+1);
    w.work.resize(optWork+1);
    tsize = w.t.size();
    lwork = w.work.size();


    using namespace PITTS;

    const auto timer = PITTS::performance::createScopedTimer<Tensor2<Type>>(
      {{"n", "m"},{n, m}}, // arguments
      {{(2.*n*m*m-2.*m*m*m/3)*kernel_info::FMA<Type>()}, // flops
       {(2.*n*m)*kernel_info::Update<Type>() + (2.*0.5*m*m)*kernel_info::Load<Type>() + (2.*0.5*m*m)*kernel_info::Store<Type>()}} // data transfers
      );

    const auto mapM = ConstEigenMap(M);

    // try to use blocked variant
    dgeqr(&n, &m, &w.tmpA(0,0), &lda, &w.t[0], &tsize, &w.work[0], &lwork, &info);

    //EigenMap(w.R).template triangularView<Eigen::Upper>() = ConstEigenMap(w.tmpA).template triangularView<Eigen::Upper>();
//    LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'U', n, m, &w.tmpA(0,0), w.tmpA.r1(), &w.R(0,0), w.R.r1());

    for(int j = 0; j < m; j++)
      for(int i = 0; i <= j; i++)
        w.R(i,j) = w.tmpA(i,j);

    const auto nChunks = w.Q.rowChunks();

#pragma omp parallel
    {
      for(int j = 0; j < m; j++)
      {
#pragma omp for schedule(static) nowait
        for(int i = 0; i < nChunks; i++)
          w.Q.chunk(i,j) = PITTS::Chunk<Type>{};
      }
    }


    for(int i = 0; i < m; i++)
      w.Q(i,i) = 1;

    char side = 'L';
    char trans = 'N';
    int ldc = w.Q.colStrideChunks()*PITTS::Chunk<Type>::size;
    dgemqr(&side, &trans, &n, &m, &m, &w.tmpA(0,0), &lda, &w.t[0], &tsize, &w.Q(0,0), &ldc, &w.work[0], &lwork, &info);

    //std::cout << "error: " << (ConstEigenMap(M) - ConstEigenMap(w.Q) * ConstEigenMap(w.R)).norm() << "\n";
    //std::cout << "R:\n" << ConstEigenMap(w.R) << "\n";
    //std::cout << "Q^T Q:\n" << ConstEigenMap(w.Q).transpose() * ConstEigenMap(w.Q) << "\n";
  }
#endif

}


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 4 )
    throw std::invalid_argument("Requires 4 arguments (n m nIter)!");

  long long n = 0, m = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], nIter);

  using Type = double;
  PITTS::Tensor2<Type> M(n,m);
  randomize(M);
  for(int i = 0; i < std::min(n,m); i++)
    M(i,i) = 5 + 0.1*M(i,i);

  Work w;
  calculate_qr(M, w);

  WorkUnpivoted w2;
  calculate_unpivoted_qr(M, w2);
  //std::cout << "unpivoted block size: " << w2.nb << "\n";

#ifdef PITTS_DIRECT_MKL_GEMM
  WorkUnpivotedGenericLAPACK w3;
  calculate_unpivoted_generic_lapack_qr(M, w3);
#endif

  PITTS::performance::clearStatistics();

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
  {
    calculate_qr(M, w);
  }
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "QR wtime: " << wtime << std::endl;

  wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
  {
    calculate_unpivoted_qr(M, w2);
  }
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "unpivoted QR wtime: " << wtime << std::endl;

#ifdef PITTS_DIRECT_MKL_GEMM
  wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
  {
    calculate_unpivoted_generic_lapack_qr(M, w3);
  }
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "generic (e.g. TSQR) unpivoted QR wtime: " << wtime << std::endl;
#endif


  PITTS::finalize();

  return 0;
}
