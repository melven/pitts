#include <Eigen/Dense>
#include "pitts_chunk.hpp"
#include "pitts_chunk_ops.hpp"
#include <exception>
#include <charconv>
#include <iostream>
#include <memory>
#include <cmath>
#include <limits>
#include <omp.h>
//#include <mkl.h>


namespace
{
  using Chunk = PITTS::Chunk<double>;

  void applyRotation(int nChunks, int firstRow, int j, const Chunk* v, const Chunk* pdata, Chunk* pdataResult)
  {
    Chunk vTx{};
    {
      int i = firstRow;
      Chunk vTx_{};
      for(; i+1 < nChunks; i+=2)
      {
        fmadd(v[i], pdata[i+nChunks*j], vTx);
        fmadd(v[i+1], pdata[i+1+nChunks*j], vTx_);
      }
      fmadd(1., vTx_, vTx);
      for(; i < nChunks; i++)
        fmadd(v[i], pdata[i+nChunks*j], vTx);
    }
    bcast_sum(vTx);
    for(int i = firstRow; i < nChunks; i++)
      fnmadd(vTx, v[i], pdata[i+nChunks*j], pdataResult[i+nChunks*j]);
  }

  // (I-vv^T)(I-ww^T) = I - vv^T - ww^T + v (vTw) w^T = I - v (v^T - vTw w^T) - w w^T
  void applyRotation2(int nChunks, int firstRow, int j, const Chunk* w, const Chunk* v, const Chunk &vTw, const Chunk* pdata, Chunk* pdataResult)
  {
    Chunk wTx{};
    Chunk vTx{};
    for(int i = firstRow; i < nChunks; i++)
    {
      fmadd(w[i], pdata[i+nChunks*j], wTx);
      fmadd(v[i], pdata[i+nChunks*j], vTx);
    }
    bcast_sum(wTx);
    bcast_sum(vTx);
    fnmadd(vTw, wTx, vTx);
    for(int i = firstRow; i < nChunks; i++)
    {
      fnmadd(wTx, w[i], pdata[i+nChunks*j], pdataResult[i+nChunks*j]);
      fnmadd(vTx, v[i], pdataResult[i+nChunks*j], pdataResult[i+nChunks*j]);
    }
  }

  // (I-vv^T)(I-ww^T) = I - vv^T - ww^T + v (vTw) w^T = I - v (v^T - vTw w^T) - w w^T
  void applyRotation2x2(int nChunks, int firstRow, int j, const Chunk* w, const Chunk* v, const Chunk &vTw, const Chunk* pdata, Chunk* pdataResult)
  {
    Chunk wTx{};
    Chunk vTx{};
    Chunk wTy{};
    Chunk vTy{};
    for(int i = firstRow; i < nChunks; i++)
    {
      fmadd(w[i], pdata[i+nChunks*j], wTx);
      fmadd(v[i], pdata[i+nChunks*j], vTx);
      fmadd(w[i], pdata[i+nChunks*(j+1)], wTy);
      fmadd(v[i], pdata[i+nChunks*(j+1)], vTy);
    }
    bcast_sum(wTx);
    bcast_sum(vTx);
    fnmadd(vTw, wTx, vTx);
    bcast_sum(wTy);
    bcast_sum(vTy);
    fnmadd(vTw, wTy, vTy);
    for(int i = firstRow; i < nChunks; i++)
    {
      fnmadd(wTx, w[i], pdata[i+nChunks*j], pdataResult[i+nChunks*j]);
      fnmadd(vTx, v[i], pdataResult[i+nChunks*j], pdataResult[i+nChunks*j]);
      fnmadd(wTy, w[i], pdata[i+nChunks*(j+1)], pdataResult[i+nChunks*(j+1)]);
      fnmadd(vTy, v[i], pdataResult[i+nChunks*(j+1)], pdataResult[i+nChunks*(j+1)]);
    }
  }

  void applyRotation2cols(int nChunks, int firstRow, int j, const Chunk* v, const Chunk* pdata, Chunk* pdataResult)
  {
    Chunk vTx{};
    Chunk vTy{};
    {
      int i = firstRow;
      Chunk vTx_{};
      Chunk vTy_{};
      for(; i+1 < nChunks; i+=2)
      {
        fmadd(v[i+0], pdata[i+0+nChunks*(j+0)], vTx);
        fmadd(v[i+0], pdata[i+0+nChunks*(j+1)], vTy);
        fmadd(v[i+1], pdata[i+1+nChunks*(j+0)], vTx_);
        fmadd(v[i+1], pdata[i+1+nChunks*(j+1)], vTy_);
      }
      fmadd(1., vTx_, vTx);
      fmadd(1., vTy_, vTy);
      for(; i < nChunks; i++)
      {
        fmadd(v[i], pdata[i+nChunks*(j+0)], vTx);
        fmadd(v[i], pdata[i+nChunks*(j+1)], vTy);
      }
    }
    bcast_sum(vTx);
    bcast_sum(vTy);
    for(int i = firstRow; i < nChunks; i++)
    {
      fnmadd(vTx, v[i], pdata[i+nChunks*(j+0)], pdataResult[i+nChunks*(j+0)]);
      fnmadd(vTy, v[i], pdata[i+nChunks*(j+1)], pdataResult[i+nChunks*(j+1)]);
    }
  }

  void householderQR(int nChunks, int m, const Chunk* pdataIn, Chunk* pdataResult)
  {
    int nPadded = nChunks*Chunk::size;
    Chunk buff_v[nChunks];
    Chunk buff_w[nChunks];
    Chunk* v = buff_v;
    Chunk* w = buff_w;
    const Chunk* pdata = pdataIn;
    for(int col = 0; col < std::min(m, nPadded); col++)
    {
      int firstRow = col / Chunk::size;
      int idx = col % Chunk::size;
      Chunk pivotChunk;
      masked_load_after(pdata[firstRow+nChunks*col], idx, pivotChunk);
      // Householder projection P = I - 2 v v^T
      // u = x - alpha e_1 with alpha = +- ||x||
      // v = u / ||u||
      double pivot = pdata[firstRow+nChunks*col][idx];
      Chunk uTu{};
      fmadd(pivotChunk, pivotChunk, uTu);
      for(int i = firstRow+1; i < nChunks; i++)
        fmadd(pdata[i+nChunks*col], pdata[i+nChunks*col], uTu);
      
      double uTu_sum = sum(uTu) + std::numeric_limits<double>::min();

      // add another minVal, s.t. the Householder reflection is correctly set up even for zero columns
      // (falls back to I - 2 e1 e1^T in that case)
      double alpha = std::sqrt(uTu_sum + std::numeric_limits<double>::min());
      //alpha *= (pivot == 0 ? -1. : -pivot / std::abs(pivot));
      alpha *= (pivot > 0 ? -1 : 1);

      uTu_sum -= pivot*alpha;
      pivot -= alpha;
      Chunk alphaChunk;
      index_bcast(Chunk{}, idx, alpha, alphaChunk);
      if( col+1 < m )
      {
        double beta = 1/std::sqrt(uTu_sum);
        fmadd(-1., alphaChunk, pivotChunk);
        mul(beta, pivotChunk, v[firstRow]);
        for(int i = firstRow+1; i < nChunks; i++)
          mul(beta, pdata[i+nChunks*col], v[i]);
      }

      // apply I - 2 v v^T     (the factor 2 is already included in v)
      // we already know column col
      masked_store_after(alphaChunk, idx, pdataResult[firstRow+nChunks*col]);
      for(int i = firstRow+1; i < nChunks; i++)
        pdataResult[i+nChunks*col] = Chunk{};
/*
{
  for(int j = col+1; j < m; j++)
  {
    Chunk vTx{};
    for(int i = firstRow; i < nChunks; i++)
      fmadd(v[i], pdataResult[i+nChunks*j], vTx);
    bcast_sum(vTx);
    for(int i = firstRow; i < nChunks; i++)
      fnmadd(vTx, v[i], pdata[i+nChunks*j], pdataResult[i+nChunks*j]);
  }
  pdata = pdataResult;
  continue;
}
*/

      // outer loop unroll (v and previous v in w)
      if( col % 2 == 1 )
      {
        if( col == 1 )
          pdata = pdataIn;

        // (I-vv^T)(I-ww^T) = I - vv^T - ww^T + v (vTw) w^T = I - v (v^T - vTw w^T) - w w^T
        Chunk vTw{};
        for(int i = firstRow; i < nChunks; i++)
          fmadd(v[i], w[i], vTw);
        bcast_sum(vTw);

        int j = col+1;
        for(; j+1 < m; j+=2)
          applyRotation2x2(nChunks, firstRow, j, w, v, vTw, pdata, pdataResult);

        for(; j < m; j++)
          applyRotation2(nChunks, firstRow, j, w, v, vTw, pdata, pdataResult);
      }
      else if( col+1 < m )
      {
        applyRotation(nChunks, firstRow, col+1, v, pdata, pdataResult);
      }

      pdata = pdataResult;
      std::swap(v,w);
    }
  }
}

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

  // make nIter a multiple of nThreads*reductionFactor
  nIter = (reductionFactor*nThreads) * ((nIter-1)/(reductionFactor*nThreads)+1);

  std::cout << "Block dimensions: " << n << " x " << m << "\n";
  std::cout << "Blocks: " << nIter << "\n";
  std::cout << "Iterations: " << nOuter << "\n";
  std::cout << "OpenMP Threads: " << nThreads << "\n";


  std::unique_ptr<Chunk[]> pdataLarge{new Chunk[nChunks*nIter*m]};

  std::unique_ptr<Chunk[]> psharedBuff(new Chunk[mChunks*nThreads*m]);

double wtime = omp_get_wtime();

#pragma omp parallel
{
  std::unique_ptr<Chunk[]> pdataSmall{new Chunk[nChunks*m]};
  std::unique_ptr<Chunk[]> plocalBuff{new Chunk[nChunks*m]};
  {
    Eigen::Map<mat> M((double*)pdataSmall.get(), nPadded, m);
    M = mat::Random(nPadded, m);
    //if( m > 1 )
    //  M.col(1) = 0.01 * mat::Zero(nPadded, 1);
    //std::cout << "Random:\n" << M << "\n";
    //Eigen::BDCSVD<mat> svd(M);
    //std::cout << "singular values:\n" << svd.singularValues().transpose() << "\n";
  }

#pragma omp for schedule(static)
  for(int iter = 0; iter < nIter; iter++)
    for(int j = 0; j < m; j++)
      for(int i = 0; i < nChunks; i++)
        for(int k = 0; k < PITTS::ALIGNMENT/64; k++)
          _mm512_store_pd(&pdataLarge[i+j*nChunks + (m*nChunks)*iter][8*k], _mm512_load_pd(&pdataSmall[j*nChunks+i][8*k]));

  //Eigen::HouseholderQR<mat> qr(nPadded, m);

for(int i = 0; i < nOuter; i++)
{
  // fill with zero
  for(int i = 0; i < nChunks; i++)
    for(int j = 0; j < m; j++)
      plocalBuff[i+nChunks*j] = Chunk{};

#pragma omp for schedule(static)
  for(int iter = 0; iter < nIter; iter++)
  {
    householderQR(nChunks, m, &pdataLarge[ (m*nChunks)*iter ], &pdataSmall[0]);

    // copy to local buffer
    for(int j = 0; j < m; j++)
      for(int i = 0; i < mChunks; i++)
        plocalBuff[(1+iter%reductionFactor)*mChunks + i + nChunks*j] = pdataSmall[i + nChunks*j];

    if( (iter+1)%reductionFactor == 0 )
      householderQR(nChunks, m, &plocalBuff[0], &plocalBuff[0]);
  }
}
int offset = omp_get_thread_num()*mChunks;
for(int j = 0; j < m; j++)
  for(int i = 0; i < mChunks; i++)
    psharedBuff[offset + i + nThreads*mChunks*j] = plocalBuff[i + nChunks*j];

} // omp parallel
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
    // calculate original matrix (data is transposed in blocks)
    mat Morig(nPadded*nIter, m);
    for(int iter = 0; iter < nIter; iter++)
    {
      Eigen::Map<mat> M(&pdataLarge[(m*nChunks)*iter][0], nPadded, m);
      Morig.block(nPadded*iter, 0, nPadded, m) = M;
    }
    {
      Eigen::HouseholderQR<mat> qr(Morig.rows(), Morig.cols());
      wtime = omp_get_wtime();
      for(int i = 0; i < nOuter; i++)
        qr.compute(Morig);
      wtime = omp_get_wtime() - wtime;
      std::cout << "ref QR wtime: " << wtime << "\n";
    }
    Eigen::BDCSVD<mat> svd(Morig.rows(), Morig.cols());
    {
      wtime = omp_get_wtime();
      for(int i = 0; i < nOuter; i++)
        svd.compute(Morig);
      wtime = omp_get_wtime() - wtime;
      std::cout << "ref SVD wtime: " << wtime << "\n";
    }
    std::cout << "singular values (ref):\n" << svd.singularValues().transpose() << "\n";
  }
*/

  return 0;
}

