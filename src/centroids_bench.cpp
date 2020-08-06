#include <mpi.h>
#include <omp.h>
#include <iostream>
#include "pitts_multivector.hpp"
#include "pitts_multivector_centroids.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_common.hpp"


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using Type = float;
  const int n = 50000, m = 50000;
  PITTS::MultiVector<Type> X(n, m), Y(n,2);
  randomize(X);

  std::vector<long long> idx(m);
  for(int j = 0; j < m; j++)
    idx[j] = j % 2;
  std::vector<float> w(m, 0.5*m);


  centroids(X, idx, w, Y);
  centroids(X, idx, w, Y);

  const auto nIter = 20;

  double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
    centroids(X, idx, w, Y);
  wtime = (omp_get_wtime() - wtime) / nIter;
  std::cout << "wtime: " << wtime << std::endl;

  PITTS::finalize();

  return 0;
}
