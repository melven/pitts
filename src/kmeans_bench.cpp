#include <mpi.h>
#include <omp.h>
#include <iostream>
#include "pitts_tensor2.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_centroids.hpp"
#include "pitts_multivector_cdist.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_performance.hpp"


int main(int argc, char* argv[])
{
#pragma omp parallel
  {
    if( omp_get_thread_num() == 0 )
      std::cout << "OpenMP #threads: " << omp_get_num_threads() << std::endl;
  }

  if( MPI_Init(&argc, &argv) != 0 )
    throw std::runtime_error("MPI error");

  int nProcs = 1, iProc = 0;
  if( MPI_Comm_size(MPI_COMM_WORLD, &nProcs) != 0 )
    throw std::runtime_error("MPI error");
  if( MPI_Comm_rank(MPI_COMM_WORLD, &iProc) != 0 )
    throw std::runtime_error("MPI error");
  if( iProc == 0 )
    std::cout << "MPI #procs: " << nProcs << std::endl;

  double wtime = omp_get_wtime();

  using Type = float;
  const int n = 50000, m = 50000/nProcs;
  PITTS::MultiVector<Type> X(n, m), Y(n,2);
  PITTS::Tensor2<Type> D(m,2);

  wtime = omp_get_wtime() - wtime;
  if( iProc == 0 )
    std::cout << "Allocated data, wtime: " << wtime << "\n";

  wtime = omp_get_wtime();

  randomize(X);

  wtime = omp_get_wtime() - wtime;
  if( iProc == 0 )
    std::cout << "Randomized data, wtime: " << wtime << "\n";

  wtime = omp_get_wtime();

  std::vector<int> idx(m);
  std::vector<float> w(m);
  float tmp_w0 = 0, tmp_w1 = 0;
#pragma omp parallel for schedule(static) reduction(+:tmp_w0,tmp_w1)
  for(int j = 0; j < m; j++)
  {
    idx[j] = j % 2;
    if( idx[j] == 0 )
      tmp_w0++;
    else
      tmp_w1++;
  }
  MPI_Allreduce(MPI_IN_PLACE, &tmp_w0, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tmp_w1, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for(int j = 0; j < m; j++)
      w[j] = 1. / (idx[j] == 0 ? tmp_w0 : tmp_w1);
  centroids(X, idx, w, Y);
  MPI_Allreduce(MPI_IN_PLACE, &Y(0,0), Y.totalPaddedSize(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  wtime = omp_get_wtime() - wtime;
  if( iProc == 0 )
    std::cout << "Initialized k-means, wtime: " << wtime << "\n";

  wtime = omp_get_wtime();

  const auto nIter = 20;

  for(int iter = 0; iter < nIter; iter++)
  {
    cdist2(X, Y, D);

    tmp_w0 = 0, tmp_w1 = 0;
#pragma omp parallel for schedule(static) reduction(+:tmp_w0,tmp_w1)
    for(int j = 0; j < m; j++)
    {
      if( D(j,0) < D(j,1) )
      {
        idx[j] = 0;
        tmp_w0++;
      }
      else
      {
        idx[j] = 1;
        tmp_w1++;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &tmp_w0, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &tmp_w1, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for(int j = 0; j < m; j++)
      w[j] = 1. / (idx[j] == 0 ? tmp_w0 : tmp_w1);

    centroids(X, idx, w, Y);
    MPI_Allreduce(MPI_IN_PLACE, &Y(0,0), Y.totalPaddedSize(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    if( iProc == 0 )
      std::cout << "Iteration: " << iter << ", w0: " << tmp_w0 << ", w1: " << tmp_w1 << "\n";
  }
  wtime = (omp_get_wtime() - wtime);
  if( iProc == 0 )
    std::cout << "20 iterations k-means wtime: " << wtime << std::endl;

  PITTS::performance::printStatistics();

  if( MPI_Finalize() != 0 )
    throw std::runtime_error("MPI error");
  return 0;
}
