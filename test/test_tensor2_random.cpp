// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensor2_random.hpp"
#include <set>

TEST(PITTS_Tensor2_random, randomize)
{
  using Tensor2_double = PITTS::Tensor2<double>;

  Tensor2_double t2(3,20);

  randomize(t2);
  EXPECT_EQ(3, t2.r1());
  EXPECT_EQ(20, t2.r2());

  // we expect different values between -1 and 1
  std::set<double> values;
  for(int i = 0; i < t2.r1(); i++)
    for(int j = 0; j < t2.r2(); j++)
        values.insert(t2(i,j));

  EXPECT_EQ(20*3, values.size());
  for(const auto& v: values)
  {
    EXPECT_LE(-1, v);
    EXPECT_LE(v, 1);
  }
}
