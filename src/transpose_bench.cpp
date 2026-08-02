// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_transpose.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_performance.hpp"
#include <iostream>
#include <charconv>
#include <stdexcept>


namespace
{
  void mkl_transpose(const PITTS::MultiVector<double>& X, PITTS::MultiVector<double>& Y)
  {
    using namespace PITTS;
    using T = double;
    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"Xrows", "Xcols"},{X.rows(),X.cols()}}, // arguments
        {{X.rows()*X.cols()*kernel_info::NoOp<T>()}, // flops
         {X.rows()*X.cols()*kernel_info::Load<T>() + X.rows()*X.cols()*kernel_info::Store<T>()}} // data transfers
        );

    Y.resize(X.cols(), X.rows());
#ifndef PITTS_DIRECT_MKL_GEMM
    EigenMap(Y).noalias() = ConstEigenMap(X).transpose();
#else
    mkl_domatcopy('c', 'T', X.rows(), X.cols(), 1., &X(0,0), X.colStrideChunks()*PITTS::Chunk<double>::size, &Y(0,0), Y.colStrideChunks()*PITTS::Chunk<double>::size);
#endif
  }
}

int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 7 )
    throw std::invalid_argument("Requires 6 arguments: n m n_ m_ reverse nIter!");

  long long n = 0, m = 0, n_ = 0, m_ = 0, nIter = 0;
  short reverse = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], n_);
  std::from_chars(argv[4], argv[5], m_);
  std::from_chars(argv[5], argv[6], reverse);
  std::from_chars(argv[6], argv[7], nIter);

  {
    const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();
    const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(n, {iProc,nProcs});
    n = nLast - nFirst + 1;
    const auto& [n_First,n_Last] = PITTS::internal::parallel::distribute(n_, {iProc,nProcs});
    n_ = n_Last - n_First + 1;
  }

  using Type = double;
  PITTS::MultiVector<Type> X(n, m), Y(n_, m_), Z(m, n);
  randomize(X);

  transpose(X, Y, {n_, m_}, bool(reverse));
  transpose(X, Y, {n_, m_}, bool(reverse));
  mkl_transpose(X, Z);
  mkl_transpose(X, Z);

  PITTS::performance::clearStatistics();

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    transpose(X, Y, {n_, m_}, bool(reverse));
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    mkl_transpose(X, Z);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "mkl wtime (without reshape): " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
