// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_random.hpp"
#include <iostream>
#include <charconv>
#include <stdexcept>


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 5 )
    throw std::invalid_argument("Requires 4 or 6 arguments (n m k nIter)!");

  long long n = 0, m = 0, k = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], k);
  std::from_chars(argv[4], argv[5], nIter);
  long long n_ = n, m_ = k;

  {
    const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();
    const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(n, {iProc,nProcs});
    n = nLast - nFirst + 1;
  }

  using Type = double;
  PITTS::MultiVector<Type> X(n, m), X_in(n,m);
  PITTS::Tensor2<Type> M(m, k);
  randomize(X);
  copy(X, X_in);
  transform(X, M);
  randomize(M);

  PITTS::performance::clearStatistics();

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
  {
    //copy(X_in, X);
    transform(X, M);
  }
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  /*
  double wtime_copy = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    copy(X_in, X);
  wtime_copy = (omp_get_wtime() - wtime_copy) / nIter;
  std::cout << "wtime_copy: " << wtime_copy << std::endl;

  std::cout << "wtime without copy: " << wtime - wtime_copy << std::endl;
  */

  PITTS::finalize();

  return 0;
}
