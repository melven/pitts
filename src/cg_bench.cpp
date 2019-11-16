#include <mpi.h>
#include <omp.h>
#include <iostream>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_laplace_operator.hpp"


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
  // there is an error for n > 16 somehow...
  PITTS::TensorTrain<Type> rhs(3,16);
  rhs.setOnes();

  // simple CG algorithm
  PITTS::TensorTrain<Type> x(rhs.dimensions);
  x.setZero();
  auto xnorm = Type(0);
  PITTS::TensorTrain<Type> r = rhs;
  auto rnorm = normalize(r);
  PITTS::TensorTrain<Type> p = r;
  auto pnorm = rnorm;
  PITTS::TensorTrain<Type> q(p.dimensions);
  const auto maxIter = 100;
  const auto resTol = 1.e-8;
  for(int iter = 0; iter < maxIter; iter++)
  {
    // q = Ap
    q.editableSubTensors() = p.subTensors(); const auto qnorm = -pnorm * laplaceOperator(q);

    // p^TAp = q^Tp
    const auto qTp = qnorm * pnorm * dot(q,p);

    // alpha = rTr / qTp
    const auto alpha = rnorm*rnorm / qTp;

    // x = x + alpha*p
    xnorm = axpby(alpha*pnorm, p, xnorm, x);

    // r = r - alpha*q
    const auto old_rnorm = rnorm;
    rnorm = axpby(-alpha*qnorm, q, rnorm, r);
    std::cout << "Residual norm: " << rnorm << std::endl;
    if( rnorm < resTol )
      break;

    // beta = rTr / old_rTr
    const auto beta = rnorm*rnorm / (old_rnorm*old_rnorm);

    pnorm = axpby(rnorm, r, beta*pnorm, p);
  }

  if( MPI_Finalize() != 0 )
    throw std::runtime_error("MPI error");
  return 0;
}
