// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_eigen.hpp"
#include "pitts_parallel.hpp"
#include "pitts_performance.hpp"
#include "pitts_common.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_triangular_solve.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include <iostream>
#include <charconv>
#include <stdexcept>


namespace
{
  using namespace PITTS;
  template<typename T>
  void trsm(MultiVector<T>& X, const Tensor2<T>& R)
  {
    // check dimensions
    if( X.cols() != R.r1() || R.r1() != R.r2() )
      throw std::invalid_argument("trsm: dimension mismatch!");

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"Xrows", "Xcols"},{X.rows(),X.cols()}}, // arguments
        {{(0.5*X.rows()*R.r1()*R.r2())*kernel_info::FMA<T>()}, // flops
         {(0.5*R.r1()*R.r2())*kernel_info::Load<T>() +
          (double(X.rows())*X.cols())*kernel_info::Update<T>()}} // data transfers
        );
    
#ifndef PITTS_DIRECT_MKL_GEMM
    ConstEigenMap(R).template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(EigenMap(X));
#else
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, X.rows(), X.cols(), 1., &R(0,0), R.r1(), &X(0,0), X.colStrideChunks()*Chunk<T>::size);
#endif
  }

  template<typename T>
  void lapmt(MultiVector<T>& X, const std::vector<int>& colPermutation)
  {
    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"Xrows", "Xcols", "XCols_new"},{X.rows(),X.cols(),int(colPermutation.size())}}, // arguments
        {{0.*kernel_info::FMA<T>()}, // flops
         {(double(X.rows())*colPermutation.size())*kernel_info::Update<T>()}} // data transfers (idealized)
        );

#ifndef PITTS_DIRECT_MKL_GEMM
    EigenMap(X) = ConstEigenMap(X)(Eigen::placeholders::all, colPermutation);
#else
    std::vector<lapack_int> jpvt(X.cols());
    for(int i = 0; i < jpvt.size(); i++)
      jpvt[i] = i < colPermutation.size() ? colPermutation[i]+1 : i+1;
    LAPACKE_dlapmt(LAPACK_COL_MAJOR, 1, X.rows(), X.cols(), &X(0,0), X.colStrideChunks()*Chunk<T>::size, jpvt.data());
#endif

    X.resize(X.rows(), colPermutation.size());
  }
}

int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 5 )
    throw std::invalid_argument("Requires 4 arguments (n m k nIter)!");

  long long n = 0, m = 0, k = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], k);
  std::from_chars(argv[4], argv[5], nIter);

  using Type = double;
  PITTS::MultiVector<Type> X_in(n, m), X(n, m), X2(n, m);
  PITTS::Tensor2<Type> M(k, k);
  randomize(X_in);
  randomize(M);
  for(int i = 0; i < k; i++)
  {
    for(int j = 0; j < k; j++)
    {
      M(j,i) = (i==j ? 1 : 0) + M(j,i) / (10*k);
      if( j > i )
        M(j,i) = 0;
    }
  }
  std::vector<int> colPermutation;

  colPermutation.resize(k);
  for(int i = 0; i < k; i++)
    colPermutation[i] = std::min<int>(i, m-1);

  copy(X_in, X);
  triangularSolve(X, M, colPermutation);
  if( m == k )
  {
    copy(X_in, X2);
    trsm(X2, M);
  }

  double copy_wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
  {
    copy(X_in, X);
  }
  copy_wtime = (omp_get_wtime() - copy_wtime) / nIter;
  std::cout << "pitts copy wtime: " << copy_wtime << std::endl;

  PITTS::performance::clearStatistics();

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
  {
    copy(X_in, X);
    triangularSolve(X, M, colPermutation);
  }
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "pitts triangularSolve wtime (without copy): " << wtime-copy_wtime << std::endl;


  if( m == k )
  {
    wtime = omp_get_wtime();
    for(int iter = 0; iter < nIter; iter++)
    {
      copy(X_in, X2);
      trsm(X2, M);
    }
    wtime = (omp_get_wtime() - wtime) / nIter;
    std::cout << "trsm wtime (without copy): " << wtime-copy_wtime << std::endl;
  }

  if( !colPermutation.empty() )
  {
    wtime = omp_get_wtime();
    for(int iter = 0; iter < nIter; iter++)
    {
      copy(X_in, X2);
      lapmt(X2, colPermutation);
      trsm(X2, M);
    }
    wtime = (omp_get_wtime() - wtime) / nIter;
    std::cout << "lapmt+trsm wtime (without copy): " << wtime-copy_wtime << std::endl;
  }

  std::cout << "Error: " << (ConstEigenMap(X) - ConstEigenMap(X2)).norm() << "\n";


  PITTS::finalize();

  return 0;
}
