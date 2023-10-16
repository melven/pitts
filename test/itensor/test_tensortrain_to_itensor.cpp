// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

// workaround ITensor / Eigen LAPACK definition problems
#ifdef EIGEN_USE_LAPACKE
#undef EIGEN_USE_LAPACKE
#endif

#include "pitts_tensortrain_to_itensor.hpp"
#include "pitts_tensortrain_from_itensor.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include <itensor/all.h>
#include <iostream>


TEST(PITTS_TensorTrain_toITensor, 2x3x4x5_random_fromToITensor)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::vector<int> dims{2,3,4,5};
  const std::vector<int> ranks{1,3,2};

  auto tt = TensorTrain_double(dims);
  tt.setTTranks(ranks);

  randomize(tt);

  const auto mps = toITensor(tt);

  ASSERT_EQ(dims.size(), length(mps));
  ASSERT_EQ(dims[0], dim(siteIndex(mps, 1)));
  ASSERT_EQ(dims[1], dim(siteIndex(mps, 2)));
  ASSERT_EQ(dims[2], dim(siteIndex(mps, 3)));
  ASSERT_EQ(dims[3], dim(siteIndex(mps, 4)));
  ASSERT_EQ(ranks.size(), length(mps)-1);
  ASSERT_EQ(ranks[0], dim(linkIndex(mps, 1)));
  ASSERT_EQ(ranks[1], dim(linkIndex(mps, 2)));
  ASSERT_EQ(ranks[2], dim(linkIndex(mps, 3)));

  auto tt_from_mps = PITTS::fromITensor<double>(mps);
  ASSERT_NEAR(0., axpby(1., tt, -1., tt_from_mps), eps);
}

