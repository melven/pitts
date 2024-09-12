// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include "pitts_mkl.hpp"
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_random.hpp"
#include <iostream>


int main(int argc, char* argv[])
{
  PITTS::initialize(&argc, &argv);

  using Type = double;
  PITTS::TensorTrain<Type> TT1(10,50);
  const int r = 750;
  TT1.setTTranks({50,r,r,r,r,r,r,r,50});
  randomize(TT1);
  rightNormalize(TT1, Type(0));
  Type tmp = 0;

  PITTS::performance::clearStatistics();

  for(int iter = 0; iter < 10; iter++)
  {
    tmp += leftNormalize(TT1);
  }
  std::cout << "random: " << tmp << std::endl;

  PITTS::finalize();

  return 0;
}
