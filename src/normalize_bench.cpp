#include <mpi.h>
#include <omp.h>
#include <iostream>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_timer.hpp"


int main(int argc, char* argv[])
{
  if( MPI_Init(&argc, &argv) != 0 )
    throw std::runtime_error("MPI error");

  int nProcs, iProc;
  if( MPI_Comm_size(MPI_COMM_WORLD, &nProcs) != 0 )
    throw std::runtime_error("MPI error");
  if( MPI_Comm_rank(MPI_COMM_WORLD, &iProc) != 0 )
    throw std::runtime_error("MPI error");
  if( iProc == 0 )
    std::cout << "MPI #procs: " << nProcs << std::endl;
#pragma omp parallel
  {
    if( iProc == 0 && omp_get_thread_num() == 0 )
      std::cout << "OpenMP #threads: " << omp_get_num_threads() << std::endl;
  }

  using Type = double;
  PITTS::TensorTrain<Type> TT1(10,100);
  const int r = 20;
  TT1.setTTranks({r,r,r,r,r,r,r,r,r});
  randomize(TT1);
  Type tmp = 0;
  for(int iter = 0; iter < 1000; iter++)
  {
    tmp += normalize(TT1);
  }
  std::cout << "random: " << tmp << std::endl;

  PITTS::timing::printStatistics();

  if( MPI_Finalize() != 0 )
    throw std::runtime_error("MPI error");
  return 0;
}
