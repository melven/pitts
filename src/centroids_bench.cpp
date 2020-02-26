#include <mpi.h>
#include <omp.h>
#include <iostream>
#include "pitts_multivector.hpp"
#include "pitts_multivector_centroids.hpp"
#include "pitts_multivector_random.hpp"


int main(int argc, char* argv[])
{
#pragma omp parallel
  {
    if( omp_get_thread_num() == 0 )
      std::cout << "OpenMP #threads: " << omp_get_num_threads() << std::endl;
  }

  if( MPI_Init(&argc, &argv) != 0 )
    throw std::runtime_error("MPI error");

  int nProcs, iProc;
  if( MPI_Comm_size(MPI_COMM_WORLD, &nProcs) != 0 )
    throw std::runtime_error("MPI error");
  if( MPI_Comm_rank(MPI_COMM_WORLD, &iProc) != 0 )
    throw std::runtime_error("MPI error");
  if( iProc == 0 )
    std::cout << "MPI #procs: " << nProcs << std::endl;

  using Type = float;
  const int n = 50000, m = 50000;
  PITTS::MultiVector<Type> X(n, m), Y(n,2);
  randomize(X);

  std::vector<int> idx(m);
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

  if( MPI_Finalize() != 0 )
    throw std::runtime_error("MPI error");
  return 0;
}
