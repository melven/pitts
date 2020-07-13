#include <Eigen/Dense>
#include "pitts_chunk.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_multivector_tsqr.hpp"
#include <exception>
#include <charconv>
#include <iostream>
#include <memory>
#include <cmath>
#include <limits>
#include <omp.h>
//#include <mkl.h>



int main(int argc, char* argv[])
{
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


  std::unique_ptr<Chunk[]> pdataLarge{new Chunk[nChunks*nIter*m]};

#pragma omp parallel
{
  std::unique_ptr<Chunk[]> pdataSmall{new Chunk[nChunks*m]};
  std::unique_ptr<Chunk[]> plocalBuff{new Chunk[nChunks*m]};
  Eigen::Map<mat> M(&pdataSmall[0][0], nPadded, m);
  M = mat::Random(nPadded, m);
  Eigen::Map<mat> Mlarge(&pdataLarge[0][0], nPadded*nIter, m);

#pragma omp for schedule(static)
  for(int iter = 0; iter < nIter; iter++)
    Mlarge.block(iter*nPadded, 0, nPadded, m) = M;
}

  //Eigen::HouseholderQR<mat> qr(nPadded, m);

double wtime = omp_get_wtime();

  std::unique_ptr<Chunk[]> psharedBuff(new Chunk[mChunks*nThreads*m]);

for(int i = 0; i < nOuter; i++)
{

#pragma omp parallel
{
  std::unique_ptr<Chunk[]> pdataSmall{new Chunk[nChunks*m]};
  std::unique_ptr<Chunk[]> plocalBuff{new Chunk[nChunks*m]};

  // fill with zero
  for(int i = 0; i < nChunks; i++)
    for(int j = 0; j < m; j++)
      plocalBuff[i+nChunks*j] = Chunk{};

  // index to the next free block in plocalBuff
  int localBuffOffset = 0;

#pragma omp for schedule(static)
  for(int iter = 0; iter < nIter; iter++)
  {
    PITTS::internal::HouseholderQR::transformBlock(nChunks, m, &pdataLarge[ nChunks*iter ], nChunks*nIter, &pdataSmall[0]);

    // copy to local buffer
    for(int j = 0; j < m; j++)
      for(int i = 0; i < mChunks; i++)
        plocalBuff[localBuffOffset + i + nChunks*j] = pdataSmall[i + nChunks*j];
    localBuffOffset += mChunks;

    if( localBuffOffset+mChunks > nChunks )
    {
      PITTS::internal::HouseholderQR::transformBlock(nChunks, m, &plocalBuff[0], nChunks, &plocalBuff[0]);
      localBuffOffset = mChunks;
    }
  }
// check if we need an additional reduction of plocalBuff
if( localBuffOffset > mChunks )
  PITTS::internal::HouseholderQR::transformBlock(nChunks, m, &plocalBuff[0], nChunks, &plocalBuff[0]);

int offset = omp_get_thread_num()*mChunks;
for(int j = 0; j < m; j++)
  for(int i = 0; i < mChunks; i++)
    psharedBuff[offset + i + nThreads*mChunks*j] = plocalBuff[i + nChunks*j];
} // omp parallel

}
wtime = omp_get_wtime() - wtime;
std::cout << "wtime: " << wtime << "\n";

  {
    Eigen::Map<mat> M((double*)psharedBuff.get(), mChunks*Chunk::size*nThreads, m);
    Eigen::BDCSVD<mat> svd(M);
    //std::cout << "Result:\n" << M << "\n";
    std::cout << "singular values (new):\n" << svd.singularValues().transpose() << "\n";
  }
/*
  {
    Eigen::Map<mat> Mlarge(&pdataLarge[0][0], nPadded*nIter, m);
    {
      Eigen::HouseholderQR<mat> qr(Mlarge.rows(), Mlarge.cols());
      wtime = omp_get_wtime();
      for(int i = 0; i < nOuter; i++)
        qr.compute(Mlarge);
      wtime = omp_get_wtime() - wtime;
      std::cout << "ref QR wtime: " << wtime << "\n";
    }
    Eigen::BDCSVD<mat> svd(Mlarge.rows(), Mlarge.cols());
    {
      wtime = omp_get_wtime();
      for(int i = 0; i < nOuter; i++)
        svd.compute(Mlarge);
      wtime = omp_get_wtime() - wtime;
      std::cout << "ref SVD wtime: " << wtime << "\n";
    }
    std::cout << "singular values (ref):\n" << svd.singularValues().transpose() << "\n";
  }
*/

  return 0;
}

