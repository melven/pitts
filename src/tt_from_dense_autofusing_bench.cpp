// Copyright (c) 2025 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_from_dense.hpp"
#include <charconv>
#include <vector>
#include <stdexcept>
#include <iostream>


// anonymous namespace for helper functions
namespace
{
  std::vector<int> fuseDims(const std::vector<int>& shape, int max_r, double computationalIntensity)
  {
    const int d = shape.size();
    std::vector<int> newShape;
    newShape.reserve(d);
    if( shape.empty() )
      return newShape;
    newShape.push_back(shape.front());
    int last_r = 1;
    // fuse dimensions by minimizing the number of flops or data transfers
    for(int i = 1; i < d; i++)
    {
      int ni = shape[i];
      int last_rn = last_r * newShape.back();
      if( last_rn < max_r )
      {
        std::cout << "fused iter " << i << ", smaller than max_r!\n";
        // first dim fused until rank is reached
        newShape.back() *= ni;
        continue;
      }
      double f = max_r * 1. / last_rn;
      double f_next = max_r * 1. / (max_r*ni);
      double f_fused = max_r * 1. / (last_rn*ni);
      double ci = computationalIntensity;

      double scaled_time = std::max(2.*last_rn, 8.*ci) + std::max(2.*f*max_r*ni, 8.*f*ci) + std::max(2.*max_r, 8.*(1.+f)*ci) + std::max(2.*f*max_r, 8.*f*(1.+f_next)*ci);
      double scaled_time_fused = std::max(2.*last_rn*ni, 8.*ci) + std::max(2.*max_r, 8.*(1.+f_fused)*ci);


      if( scaled_time_fused < 0.9 * scaled_time )
      {
        std::cout << "FUSED iter: " << i << ", f: " << f << ", f_next: " << f_next << ", f_fused: " << f_fused << ", ci: " << ci << ", scaled_time: " << scaled_time << ", scaled_time_fused: " << scaled_time_fused << "\n";
        newShape.back() *= ni;
        continue;
      }
        
      std::cout << "not fused iter: " << i << ", f: " << f << ", f_next: " << f_next << ", f_fused: " << f_fused << ", ci: " << ci << ", scaled_time: " << scaled_time << ", scaled_time_fused: " << scaled_time_fused << "\n";

      // new dim
      newShape.push_back(ni);
      last_r = max_r;
    }

    return newShape;
  }
}


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc < 5 || argc > 7 )
    throw std::invalid_argument("Requires 4 to 6 arguments (n d max_r nIter [localDims] [computationalIntensity])!");

  std::size_t n = 0, d = 0, max_r = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], d);
  std::from_chars(argv[3], argv[4], max_r);
  std::from_chars(argv[4], argv[5], nIter);
  int localDims = 0;
  if( argc >= 6 )
    std::from_chars(argv[5], argv[6], localDims);
  double computationalIntensity = 12.;
  if( argc >= 7 )
    std::from_chars(argv[6], argv[7], computationalIntensity);

  const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();

  // compress shape, s.t. we optimize the reduction factor
  // first dimension is distributed over MPI processes
  std::vector<int> reversedShape(d,n);
  if( localDims )
    reversedShape.push_back(nProcs*localDims);
  // setup optimized shape with fused dimensions for the calculation
  std::vector<int> newReversedShape = fuseDims(reversedShape, max_r, computationalIntensity);

  // fuse last dimensions until we do not have operations with less rows than columns
  while( newReversedShape.size() > 2 && newReversedShape.back() < max_r )
  {
    int last_n = newReversedShape.back();
    newReversedShape.pop_back();
    newReversedShape.back() *= last_n;
  }

  std::vector<int> newShape(newReversedShape.rbegin(), newReversedShape.rend());
  if( iProc == 0 )
  {
    std::cout << "Called tt_fromDense_autofusing_bench. Dimensions: " << n << "^" << d << ", max rank: " << max_r << ", localDims: " << localDims << "\n";
    std::cout << "fused dimensions:";
    for(auto ni: newShape)
      std::cout << "  " << ni;
  }
  std::cout << "\n";

  // distribute first dimension
  {
    const auto& [first,last] = PITTS::internal::parallel::distribute(newShape[0], {iProc,nProcs});
    newShape[0] = last - first + 1;
  }

  std::size_t nTotal = 1;
  for(auto ni: newShape)
    nTotal *= ni;


  PITTS::MultiVector<double> X(nTotal/newShape.back(), newShape.back());
  PITTS::MultiVector<double> work(nTotal, 1);
  randomize(work);


  double min_wtime = std::numeric_limits<double>::max();
  for(int iter = 0; iter < nIter; iter++)
  {
    X.resize(nTotal/newShape.back(), newShape.back());
    randomize(X);
    double wtime = omp_get_wtime();

    const auto TT = fromDense(X, work, newShape, 1.e-8, max_r, true);

    wtime = omp_get_wtime() - wtime;
    if( iProc == 0 )
      std::cout << "wtime: " << wtime << "\n";
    min_wtime = std::min(min_wtime, wtime);
  }
  if( iProc == 0 )
    std::cout << "min. wtime: " << min_wtime << "\n";

  PITTS::finalize();

  return 0;
}

