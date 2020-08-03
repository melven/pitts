#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <charconv>
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_common.hpp"


int main(int argc, char* argv[])
{
//  PITTS::initialize(&argc, &argv);

  if( argc != 4 )
    throw std::invalid_argument("Requires 3!");

  std::size_t n = 0, m = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], nIter);

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
