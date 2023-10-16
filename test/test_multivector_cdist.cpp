// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_multivector_cdist.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"

TEST(PITTS_MultiVector_cdist, single_vector)
{
  constexpr auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(50,1), Y(50,1);
  Tensor2_double D(1,1);

  randomize(X);
  randomize(Y);

  cdist2(X, Y, D);

  double ref = 0;
  for(int i = 0; i < 50; i++)
    ref += (X(i,0)-Y(i,0)) * (X(i,0)-Y(i,0));

  ASSERT_EQ(1, D.r1());
  ASSERT_EQ(1, D.r2());

  EXPECT_NEAR(ref, D(0,0), eps);
}


TEST(PITTS_MultiVector_cdist, vec3_x_vec2)
{
  constexpr auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(20,3), Y(20,2);
  Tensor2_double D(3,2);

  randomize(X);
  randomize(Y);

  cdist2(X, Y, D);

  ASSERT_EQ(3, D.r1());
  ASSERT_EQ(2, D.r2());

  for(int i = 0; i < 3; i++)
  {
    for(int j = 0; j < 2; j++)
    {
      double ref = 0;
      for(int k = 0; k < 20; k++)
        ref += (X(k,i)-Y(k,j)) * (X(k,i)-Y(k,j));
      EXPECT_NEAR(ref, D(i,j), eps);
    }
  }
}
