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

  if( argc != 5 && argc != 7 )
    throw std::invalid_argument("Requires 4 or 6 arguments (n m k nIter n_ m_)!");

  long long n = 0, m = 0, k = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], k);
  std::from_chars(argv[4], argv[5], nIter);
  long long n_ = n, m_ = k;
  if( argc == 7 )
  {
    std::from_chars(argv[5], argv[6], n_);
    std::from_chars(argv[6], argv[7], m_);
  }

  {
    const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();
    const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(n, {iProc,nProcs});
    n = nLast - nFirst + 1;
    const auto& [n_First,n_Last] = PITTS::internal::parallel::distribute(n_, {iProc,nProcs});
    n_ = n_Last - n_First + 1;
  }

  using Type = double;
  PITTS::MultiVector<Type> X(n, m), Y(n_, m_);
  PITTS::Tensor2<Type> M(m, k);
  randomize(X);
  randomize(M);

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    transform(X, M, Y, {n_, m_});
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
