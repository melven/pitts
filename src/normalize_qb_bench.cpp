#include "omp.h"
#include "pitts_mkl.hpp"
#include "pitts_common.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor3_split.hpp"
#include <iostream>
#include <charconv>


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc != 5 )
    throw std::invalid_argument("Requires 4 arguments (n m nIter leftOrthog)!");

  long long n = 0, m = 0, nIter = 0, leftOrthog = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], nIter);
  std::from_chars(argv[3], argv[4], leftOrthog);

  using Type = double;
  PITTS::Tensor2<Type> M(n, m);
  randomize(M);

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    PITTS::internal::normalize_qb(M, leftOrthog);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  PITTS::finalize(true);

  return 0;
}
