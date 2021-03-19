#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include <exception>
#include <charconv>
#include <iostream>
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Chunk = PITTS::Chunk<double>;

  if( argc != 5 )
    throw std::invalid_argument("Requires 4 arguments (n m reductionFactor nIter)!");

  long long n = 0, m = 0;
  int reductionFactor = 20, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], reductionFactor);
  std::from_chars(argv[4], argv[5], nIter);

  const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();
  {
    const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(n, {iProc,nProcs});
    n = nLast - nFirst + 1;
  }

  PITTS::MultiVector<double> M(n, m);
  randomize(M);

  PITTS::Tensor2<double> R(m,m);

double wtime = omp_get_wtime();
  for(int iter = 0; iter < nIter; iter++)
  {
    block_TSQR(M, R, reductionFactor);
  }
wtime = omp_get_wtime() - wtime;
  if( iProc == 0 )
    std::cout << "wtime: " << wtime << "\n";

  if( iProc == 0 )
  {
    Eigen::BDCSVD<mat> svd(ConstEigenMap(R));
    //std::cout << "Result:\n" << M << "\n";
    std::cout << "singular values (new):\n" << svd.singularValues().transpose() << "\n";
  }

  PITTS::finalize();

  return 0;
}

