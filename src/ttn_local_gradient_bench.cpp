// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_common.hpp"
#include "pitts_ttn_local_gradient_contraction.hpp"
#include "pitts_parallel.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_tensor3_random.hpp"
#include <iostream>
#include <charconv>
#include <stdexcept>


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 7 )
    throw std::invalid_argument("Requires 6 arguments (b_left b_right b_top n_classes n_samples nIter)!");

  long long b_left = 0, b_right = 0, b_top = 0, n_classes = 0, n_samples = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], b_left);
  std::from_chars(argv[2], argv[3], b_right);
  std::from_chars(argv[3], argv[4], b_top);
  std::from_chars(argv[4], argv[5], n_classes);
  std::from_chars(argv[5], argv[6], n_samples);
  std::from_chars(argv[6], argv[7], nIter);

  bool mpiParallel = false;
  {
    const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();
    const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(n_samples, {iProc,nProcs});
    n_samples = nLast - nFirst + 1;
    if( nProcs > 0 )
      mpiParallel = true;
  }

  using Type = double;
  PITTS::Tensor3<Type> optTensor(b_left, b_right, b_top);
  PITTS::Tensor3<Type> gradOptTensor(b_left, b_right, b_top);
  PITTS::MultiVector<Type> envLeft(n_samples, b_left), envRight(n_samples, b_right), envTop(n_samples, b_top * n_classes);

  randomize(optTensor);
  randomize(gradOptTensor);
  randomize(envLeft);
  randomize(envRight);
  randomize(envTop);

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    ttn_local_gradient_contract(optTensor, envLeft, envRight, envTop, gradOptTensor, mpiParallel);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
