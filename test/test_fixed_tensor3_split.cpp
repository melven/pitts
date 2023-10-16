// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_fixed_tensor3_split.hpp"
#include "pitts_fixed_tensor3_combine.hpp"
#include <complex>

template<typename T>
class PITTS_FixedTensor3_split : public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_CASE(PITTS_FixedTensor3_split, TestTypes);

TYPED_TEST(PITTS_FixedTensor3_split, N_equals_one)
{
  using Type = typename TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,1>;
  constexpr auto eps = 1.e-10;

  FixedTensor3 t3c(5,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      t3c(i,0,j) = 100 + i*10 + j;

  FixedTensor3 t3a, t3b;
  split(t3c, t3a, t3b);
  const auto t3c_ = combine(t3a, t3b);

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
    {
      EXPECT_NEAR(t3c(i,0,j), t3c_(i,0,j), eps);
    }
}

TYPED_TEST(PITTS_FixedTensor3_split, N_equals_four)
{
  using Type = typename TestFixture::Type;
  using FixedTensor3_4 = PITTS::FixedTensor3<Type,4>;
  using FixedTensor3_2 = PITTS::FixedTensor3<Type,2>;
  constexpr auto eps = 1.e-10;

  FixedTensor3_4 t3c(5,3);
  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
        t3c(i,k,j) = 100 + i*10 + j + k * 0.9777;

  FixedTensor3_2 t3a, t3b;
  split(t3c, t3a, t3b);
  const auto t3c_ = combine(t3a, t3b);

  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
      {
        EXPECT_NEAR(t3c(i,k,j), t3c_(i,k,j), eps);
      }
}

TYPED_TEST(PITTS_FixedTensor3_split, N_equals_four_rightOrthog)
{
  using Type = typename TestFixture::Type;
  using FixedTensor3_4 = PITTS::FixedTensor3<Type,4>;
  using FixedTensor3_2 = PITTS::FixedTensor3<Type,2>;
  constexpr auto eps = 1.e-10;

  FixedTensor3_4 t3c(5,3);
  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
        t3c(i,k,j) = 100 + i*10 + j + k * 0.9777;

  FixedTensor3_2 t3a, t3b;
  split(t3c, t3a, t3b, false);
  const auto t3c_ = combine(t3a, t3b);

  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
      {
        EXPECT_NEAR(t3c(i,k,j), t3c_(i,k,j), eps);
      }
}
