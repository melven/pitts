#include <Eigen/Dense>
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_common.hpp"
#include "pitts_performance.hpp"
#include <exception>
#include <charconv>
#include <iostream>
#include <omp.h>



int main(int argc, char* argv[])
{
  //PITTS::initialize(&argc, &argv);

  using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Chunk = PITTS::Chunk<double>;

  if( argc != 5 )
    throw std::invalid_argument("Requires 4 arguments!");

  std::size_t n = 0, m = 0, nIter = 0, nOuter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], m);
  std::from_chars(argv[3], argv[4], nIter);
  std::from_chars(argv[4], argv[5], nOuter);

int nThreads = 1;
#pragma omp parallel shared(nThreads)
{
#pragma omp critical
  nThreads = omp_get_num_threads();
}
  std::size_t nChunks = (n-1) / Chunk::size + 1;
  std::size_t nPadded = nChunks * Chunk::size;

  std::size_t mChunks = (m-1) / Chunk::size + 1;
  int reductionFactor = nChunks / mChunks - 1;
  if( reductionFactor < 1 )
    throw std::invalid_argument("block size is too small!");

  std::cout << "Block dimensions: " << n << " x " << m << "\n";
  std::cout << "Blocks: " << nIter << "\n";
  std::cout << "Iterations: " << nOuter << "\n";
  std::cout << "OpenMP Threads: " << nThreads << "\n";

  PITTS::MultiVector<double> M(nPadded*nIter, m);
  randomize(M);

  PITTS::Tensor2<double> R(m,m);

double wtime = omp_get_wtime();
  for(int iOuter = 0; iOuter < nOuter; iOuter++)
  {
    block_TSQR(M, R, reductionFactor);
  }
wtime = omp_get_wtime() - wtime;
std::cout << "wtime: " << wtime << "\n";

  {
    Eigen::BDCSVD<mat> svd(ConstEigenMap(R));
    //std::cout << "Result:\n" << M << "\n";
    std::cout << "singular values (new):\n" << svd.singularValues().transpose() << "\n";
  }

  PITTS::performance::printStatistics();
  //PITTS::finalize();

  return 0;
}

