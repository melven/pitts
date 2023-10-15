// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include <iostream>
#include <charconv>


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 4 )
    throw std::invalid_argument("Requires 3 arguments (n m nIter)!");

  std::size_t n = 0, m = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], nIter);

  {
    const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(n, PITTS::internal::parallel::mpiProcInfo());
    n = nLast - nFirst + 1;
  }

  using Type = double;
  PITTS::MultiVector<Type> X(n, m), Y(n, m);
  randomize(X);
  randomize(Y);

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    copy(X, Y);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
