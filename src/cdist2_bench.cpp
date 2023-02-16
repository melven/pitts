#include <iostream>
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_cdist.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using Type = float;
  const int n = 50000, m = 50000;
  PITTS::MultiVector<Type> X(n, m), Y(n,2);
  randomize(X);
  randomize(Y);

  PITTS::Tensor2<Type> D(m,2);

  cdist2(X, Y, D);
  cdist2(X, Y, D);

  const auto nIter = 20;

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    cdist2(X, Y, D);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
