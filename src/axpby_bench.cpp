// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include <iostream>
#include <charconv>


int main(int argc, char* argv[])
{
  if( argc != 6 && argc != 8 )
    throw std::invalid_argument("Requires 5 or 7 arguments (n d rX rY nIter [orthoX] [orthoY])!\n  (orthoX and orthoY can be 'L', 'N', or 'R')");

  int n, d, rX, rY, nIter;
  std::from_chars(argv[1], argv[2], n);
  std::from_chars(argv[2], argv[3], d);
  std::from_chars(argv[3], argv[4], rX);
  std::from_chars(argv[4], argv[5], rY);
  std::from_chars(argv[5], argv[6], nIter);

  char orthoX = 'N', orthoY = 'N';
  if( argc == 8 )
  {
    orthoX = *argv[6];
    orthoY = *argv[7];
    std::cout << "orthoX: " << orthoX << ", orthoY: " << orthoY << "\n";
  }

  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrain<Type> TTx(d,n), TTy(d, n), TTz(d, n);
  TTx.setTTranks(rX);
  TTy.setTTranks(rY);
  randomize(TTx);
  randomize(TTy);
  if( orthoX == 'L' )
    leftNormalize(TTx, 0.);
  else if( orthoY == 'R' )
    rightNormalize(TTx, 0.);

  if( orthoY == 'L' )
    leftNormalize(TTy, 0.);
  else if( orthoY == 'R' )
    rightNormalize(TTy, 0.);

  copy(TTy, TTz);

  // first calls to MKL are often slower...
  for(int iter = 0; iter < 2; iter++)
  {
    copy(TTz, TTy);
    axpby(0.01, TTx, 0.9, TTy, Type(0), std::max(rX, rY));
  }

  PITTS::performance::clearStatistics();

  for(int iter = 0; iter < nIter; iter++)
  {
    copy(TTz, TTy);
    axpby(0.01, TTx, 0.9, TTy, Type(0), std::max(rX, rY));
  }

  PITTS::finalize();

  return 0;
}
