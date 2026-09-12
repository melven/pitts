// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_performance.hpp"
#include "pitts_random.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_cross.hpp"
#include "pitts_tensortrain_norm.hpp"
#include <charconv>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{
  auto& operator<<(auto& stream, const std::vector<int>& vec)
  {
    for(auto v: vec)
      stream << v << " ";
    return stream;
  }

  template<typename T>
  void printInfo(const std::string& name, const PITTS::TTCrossInfo<T>& info, const PITTS::TensorTrain<T>& TT)
  {
    std::cout << name << ": " << info.nSweeps << " sweeps, " << info.nEvaluations << " evaluations, "
              << "last change " << info.lastChange << ", converged " << info.converged << "\n";
    std::cout << name << " TTranks: " << TT.getTTranks() << "\n";
  }
}


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  if( argc < 4 || argc > 8 )
    throw std::invalid_argument("Requires 3 to 7 arguments (n d r [nMALS] [maxRank] [tolerance] [nSweeps])!\n"
                                "  n: dimension of each tensor index, d: number of dimensions,\n"
                                "  r: TT rank of the input, nMALS: 1 (ALS-cross) or 2 (DMRG-cross), maxRank: max. TT rank of the result,\n"
                                "  tolerance: truncation and convergence tolerance of the TT-cross,\n"
                                "  nSweeps: maximal number of full sweeps of the TT-cross");

  using Type = double;

  int n = 0, d = 0, r = 0, nMALS = 1, maxRank = 50, nSweeps = 10;
  Type tolerance = std::sqrt(std::numeric_limits<Type>::epsilon());
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], d);
  std::from_chars(argv[3], argv[4], r);
  if( argc > 4 )
    std::from_chars(argv[4], argv[5], nMALS);
  if( argc > 5 )
    std::from_chars(argv[5], argv[6], maxRank);
  if( argc > 6 )
    std::from_chars(argv[6], argv[7], tolerance);
  if( argc > 7 )
    std::from_chars(argv[7], argv[8], nSweeps);

  // random input with only positive entries, so sqrt(x*x) == x (round-trip check below);
  // the 1/r scaling of the core entries keeps the tensor entries in the order of 1/r
  // (the contraction over each bond sums up r terms)
  PITTS::TensorTrain<Type> TTx(d, n);
  TTx.setTTranks(r);
  std::uniform_real_distribution<Type> distribution(Type(0.5)/r, Type(1.5)/r);
  for(int k = 0; k < d; k++)
    TTx.editSubTensor(k, [&](PITTS::Tensor3<Type>& X)
    {
      for(int i = 0; i < X.r1(); i++)
        for(int j = 0; j < X.n(); j++)
          for(int l = 0; l < X.r2(); l++)
            X(i,j,l) = distribution(PITTS::internal::randomGenerator);
    });

  PITTS::TTCrossOptions<Type> options;
  options.nMALS = nMALS;
  options.maxRank = maxRank;
  options.rankTolerance = tolerance;
  options.nSweeps = nSweeps;
  // the error of the low-rank approximation has at most the rank of the input, so enrich by r directions
  // (with a slightly larger rank for the error approximation itself)
  options.enrichmentRank = r;
  options.errorRank = r + 5;
  options.convergenceTolerance = tolerance;
  options.verbose = true;

  PITTS::TensorTrain<Type> TTy(TTx.dimensions()), TTz(TTx.dimensions());

  PITTS::performance::clearStatistics();

  // y ~ x*x (exact TT ranks r*r) ...
  const auto infoSquare = applyElementwise([](Type v){return v*v;}, TTx, TTy, options);
  std::cout << "\n### timings for the element-wise square ###\n";
  PITTS::performance::printStatistics();

  // ... and back: z ~ sqrt(y) = x (only approximately low-rank)
  const auto infoSqrt = applyElementwise([](Type v){return std::sqrt(v);}, TTy, TTz, options);
  std::cout << "\n### timings for the element-wise sqrt ###\n";
  PITTS::performance::printStatistics();

  printInfo("x*x", infoSquare, TTy);
  printInfo("sqrt(x*x)", infoSqrt, TTz);

  // real error of the round-trip (no truncation in the axpby, so this is the exact difference)
  const auto xNorm = norm2(TTx);
  const auto errNorm = axpby(Type(-1), TTx, Type(1), TTz, Type(0));
  std::cout << "relative round-trip error ||sqrt(x*x) - x|| / ||x||: " << errNorm/xNorm << "\n";

  // don't print the statistics of the error calculation above
  PITTS::finalize(false);

  return 0;
}
