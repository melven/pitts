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
#include "pitts_tensortrain_from_dense_twosided.hpp"
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


      if( scaled_time_fused < scaled_time )
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

  if( argc < 5 || argc > 6 )
    throw std::invalid_argument("Requires 4 or 5 arguments (n d max_r nIter [computationalIntensity])!");

  std::size_t n = 0, d = 0, max_r = 0, nIter = 0;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], d);
  std::from_chars(argv[3], argv[4], max_r);
  std::from_chars(argv[4], argv[5], nIter);
  double computationalIntensity = 12.;
  if( argc >= 6 )
    std::from_chars(argv[5], argv[6], computationalIntensity);

  // compress shape, s.t. we optimize the reduction factor
  std::vector<int> shapeLeft = fuseDims(std::vector<int>(d/2, n), max_r, computationalIntensity);
  std::vector<int> shapeRight = fuseDims(std::vector<int>(d-d/2, n), max_r, computationalIntensity);
  std::vector<int> shape = shapeLeft;
  shape.insert(shape.end(), shapeRight.rbegin(), shapeRight.rend());

  std::cout << "Called tt_fromDense_twosided_autofusing_bench. Dimensions: " << n << "^" << d << ", max rank: " << max_r << "\n";
  std::cout << "fused dimensions:";
  for(auto ni: shape)
    std::cout << "  " << ni;
  std::cout << "\n";


  std::size_t nTotal = 1;
  for(auto ni: shape)
    nTotal *= ni;


  PITTS::MultiVector<double> X(nTotal/shape.back(), shape.back());
  PITTS::MultiVector<double> work(nTotal, 1);
  randomize(work);


  double min_wtime = std::numeric_limits<double>::max();
  for(int iter = 0; iter < nIter; iter++)
  {
    X.resize(nTotal/shape.back(), shape.back());
    randomize(X);
    double wtime = omp_get_wtime();

    const auto TT = fromDense_twoSided(X, work, shape, 1.e-8, max_r);

    wtime = omp_get_wtime() - wtime;
    std::cout << "wtime: " << wtime << "\n";
    min_wtime = std::min(min_wtime, wtime);
  }
  std::cout << "min. wtime: " << min_wtime << "\n";

  PITTS::finalize();

  return 0;
}

