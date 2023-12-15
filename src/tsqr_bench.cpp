// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_eigen.hpp"
#include <charconv>
#include <iostream>
#include <stdexcept>


#ifdef TYPE
using Type = TYPE;
#else
using Type = double;
#endif


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using Chunk = PITTS::Chunk<Type>;

  if( argc != 5 && argc != 6 )
    throw std::invalid_argument("Requires 4 arguments (n m reductionFactor nIter [colBlockingSize] )!");

  long long n = 0, m = 0;
  int reductionFactor = 20, nIter = 0;
  int colBlockingSize = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], reductionFactor);
  std::from_chars(argv[4], argv[5], nIter);
  if( argc == 6 )
    std::from_chars(argv[5], argv[6], colBlockingSize);

  const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();
  {
    const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(n, {iProc,nProcs});
    n = nLast - nFirst + 1;
  }

  PITTS::MultiVector<Type> M(n, m);
  randomize(M);

  PITTS::Tensor2<Type> R(m,m);

double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
  {
    block_TSQR(M, R, reductionFactor, true, colBlockingSize);
  }
wtime = omp_get_wtime() - wtime;
  if( iProc == 0 )
    std::cout << "wtime: " << wtime << "\n";

  if( iProc == 0 )
  {
    Eigen::BDCSVD<mat> svd(ConstEigenMap(R));
    //std::cout << "Result:\n" << M << "\n";
    std::cout << "singular values (new):\n" << svd.singularValues().transpose() << "\n";
  }

  PITTS::finalize();

  return 0;
}

