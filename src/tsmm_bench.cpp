#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <charconv>
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_common.hpp"


int main(int argc, char* argv[])
{
//  PITTS::initialize(&argc, &argv);

  if( argc != 5 && argc != 7 )
    throw std::invalid_argument("Requires 4 or 6 arguments!");

  std::size_t n = 0, m = 0, k = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], k);
  std::from_chars(argv[4], argv[5], nIter);
  std::size_t n_ = n, m_ = m;
  if( argc == 7 )
  {
    std::from_chars(argv[5], argv[6], n_);
    std::from_chars(argv[6], argv[7], m_);
  }


  using Type = double;
  PITTS::MultiVector<Type> X(n, m), Y(n_, m_);
  PITTS::Tensor2<Type> M(m, k);
  randomize(X);
  randomize(M);

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    transform(X, M, Y);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
