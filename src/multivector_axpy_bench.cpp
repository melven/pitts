// Copyright (c) 20204German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_axpby.hpp"
#include "pitts_multivector_dot.hpp"
#include "pitts_multivector_norm.hpp"
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
  PITTS::MultiVector<Type> X(n, m), Y(n, m), Z(n, m);
  randomize(X);
  randomize(Y);
  randomize(Z);

  using arr = Eigen::ArrayX<Type>;
  arr alpha = arr::Random(m) / nIter / 100;
  arr beta = arr::Zero(m);

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    axpy(alpha, X, Y);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "axpy wtime: " << wtime << std::endl;

  wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    beta = dot(X, Y);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "dot wtime: " << wtime << std::endl;

  wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    beta = norm2(X);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "norm wtime: " << wtime << std::endl;

  wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    beta = axpy_dot(alpha, X, Y, Z);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "axpy_dot wtime: " << wtime << std::endl;

  wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    beta = axpy_norm2(alpha, X, Y);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "axpy_norm wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
