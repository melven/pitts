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
#include "pitts_multivector_triangular_solve.hpp"
#include "pitts_multivector_random.hpp"
#include <iostream>
#include <charconv>
#include <stdexcept>


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
  PITTS::MultiVector<Type> X_in(n, m), X(n, m);
  PITTS::Tensor2<Type> M(k, k);
  randomize(X_in);
  randomize(M);
  for(int i = 0; i < k; i++)
    M(i,i) = 1 + 0.1*M(i,i);
  std::vector<int> colPermutation;
  if( m != k )
  {
    colPermutation.resize(k);
    for(int i = 0; i < k; i++)
      colPermutation[i] = m-i-1;
  }
  copy(X_in, X);
  triangularSolve(X, M, colPermutation);

  PITTS::performance::clearStatistics();

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
  {
    //copy(X_in, X);
    triangularSolve(X, M, colPermutation);
  }
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
