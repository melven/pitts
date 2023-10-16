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
#include <iostream>
#include <charconv>
#include <stdexcept>


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 7 )
    throw std::invalid_argument("Requires 5 arguments: n m n_ m_ reverse nIter!");

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
    n_ = nLast - nFirst + 1;
  }

  using Type = double;
  PITTS::MultiVector<Type> X(n, m), Y(n_, m_);
  randomize(X);

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    transpose(X, Y, {n_, m_}, bool(reverse));
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
