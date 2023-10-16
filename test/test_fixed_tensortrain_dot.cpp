// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_fixed_tensortrain_dot.hpp"
#include "pitts_fixed_tensortrain_random.hpp"
#include <complex>

template<typename T>
class PITTS_FixedTensorTrain_dot : public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_CASE(PITTS_FixedTensorTrain_dot, TestTypes);

TYPED_TEST(PITTS_FixedTensorTrain_dot, rank_1_vector_self)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,5>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT(1);

  TT.setZero();
  EXPECT_NEAR(0, dot(TT,TT), eps);

  TT.setOnes();
  EXPECT_NEAR(5., dot(TT,TT), eps);

  TT.setUnit({2});
  EXPECT_NEAR(1, dot(TT,TT), eps);
}


TYPED_TEST(PITTS_FixedTensorTrain_dot, large_rank_1_vector_self)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,50>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT(1);

  TT.setZero();
  EXPECT_NEAR(0, dot(TT,TT), eps);

  TT.setOnes();
  EXPECT_NEAR(50., dot(TT,TT), eps);

  TT.setUnit({2});
  EXPECT_NEAR(1, dot(TT,TT), eps);
}


TYPED_TEST(PITTS_FixedTensorTrain_dot, rank_1_vector_random)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,5>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT(1), unitTT(1);

  randomize(TT);
  Type tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    unitTT.setUnit({i});
    tmp += std::pow(dot(TT,unitTT),2);
  }
  EXPECT_NEAR(tmp, dot(TT,TT), eps);
}

TYPED_TEST(PITTS_FixedTensorTrain_dot, rank_1_vector_random_other)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,5>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT1(1), TT2(1), unitTT(1);

  randomize(TT1);
  randomize(TT2);
  Type tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    unitTT.setUnit({i});
    tmp += dot(TT1,unitTT) * dot(TT2,unitTT);
  }
  EXPECT_NEAR(tmp, dot(TT1,TT2), eps);
}

TYPED_TEST(PITTS_FixedTensorTrain_dot, rank_2_matrix_self)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,5>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT(2);

  TT.setZero();
  EXPECT_NEAR(0, dot(TT,TT), eps);

  TT.setOnes();
  EXPECT_NEAR(25., dot(TT,TT), eps);

  TT.setUnit({2,1});
  EXPECT_NEAR(1, dot(TT,TT), eps);
}

TYPED_TEST(PITTS_FixedTensorTrain_dot, rank_2_matrix_random_self)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,5>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT(2), unitTT(2);

  // TT-rank==1
  randomize(TT);
  Type tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 5; j++)
    {
      unitTT.setUnit({i,j});
      tmp += std::pow(dot(TT,unitTT),2);
    }
  }
  EXPECT_NEAR(tmp, dot(TT,TT), eps);

  // TT-rank==3
  TT.setTTranks(3);
  randomize(TT);
  tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 5; j++)
    {
      unitTT.setUnit({i,j});
      tmp += std::pow(dot(TT,unitTT),2);
    }
  }
  EXPECT_NEAR(tmp, dot(TT,TT), eps);
}

TYPED_TEST(PITTS_FixedTensorTrain_dot, rank_2_matrix_random_other)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,5>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT1(2), TT2(2), unitTT(2);

  // TT-rank==1
  randomize(TT1);
  randomize(TT2);
  Type tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 5; j++)
    {
      unitTT.setUnit({i,j});
      tmp += dot(TT1,unitTT) * dot(TT2,unitTT);
    }
  }
  EXPECT_NEAR(tmp, dot(TT1,TT2), eps);

  // TT-rank>1
  TT1.setTTranks(3);
  randomize(TT1);
  TT2.setTTranks(2);
  randomize(TT2);
  tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 5; j++)
    {
      unitTT.setUnit({i,j});
      tmp += dot(TT1,unitTT) * dot(TT2,unitTT);
    }
  }
  EXPECT_NEAR(tmp, dot(TT1,TT2), eps);
}


TYPED_TEST(PITTS_FixedTensorTrain_dot, rank_1_vector_unit)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,3>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT1(1), TT2(1);

  TT1.setUnit({0});
  TT2.setUnit({0});
  EXPECT_NEAR(1, dot(TT1,TT2), eps);
  EXPECT_NEAR(1, dot(TT2,TT1), eps);

  TT2.setUnit({1});
  EXPECT_NEAR(0, dot(TT1,TT2), eps);
  EXPECT_NEAR(0, dot(TT2,TT1), eps);

  TT2.setUnit({2});
  EXPECT_NEAR(0, dot(TT1,TT2), eps);
  EXPECT_NEAR(0, dot(TT2,TT1), eps);

  TT1.setUnit({2});
  EXPECT_NEAR(1, dot(TT1,TT2), eps);
  EXPECT_NEAR(1, dot(TT2,TT1), eps);
}

TYPED_TEST(PITTS_FixedTensorTrain_dot, rank_2_matrix_unit)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,3>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT1(2), TT2(2);

  TT1.setUnit({0,0});
  TT2.setUnit({0,0});
  EXPECT_NEAR(1, dot(TT1,TT2), eps);
  EXPECT_NEAR(1, dot(TT2,TT1), eps);

  TT2.setUnit({1,0});
  EXPECT_NEAR(0, dot(TT1,TT2), eps);
  EXPECT_NEAR(0, dot(TT2,TT1), eps);

  TT2.setUnit({2,0});
  EXPECT_NEAR(0, dot(TT1,TT2), eps);
  EXPECT_NEAR(0, dot(TT2,TT1), eps);

  TT1.setUnit({2,0});
  EXPECT_NEAR(1, dot(TT1,TT2), eps);
  EXPECT_NEAR(1, dot(TT2,TT1), eps);


  TT1.setUnit({0,1});
  TT2.setUnit({0,1});
  EXPECT_NEAR(1, dot(TT1,TT2), eps);
  EXPECT_NEAR(1, dot(TT2,TT1), eps);

  TT2.setUnit({1,0});
  EXPECT_NEAR(0, dot(TT1,TT2), eps);
  EXPECT_NEAR(0, dot(TT2,TT1), eps);
}

TYPED_TEST(PITTS_FixedTensorTrain_dot, rank_4_tensor_unit)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,3>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT1(4), TT2(4);

  TT1.setUnit({0,0,1,2});
  TT2.setUnit({0,0,1,2});
  EXPECT_NEAR(1, dot(TT1,TT2), eps);
  EXPECT_NEAR(1, dot(TT2,TT1), eps);

  TT2.setUnit({1,0,1,2});
  EXPECT_NEAR(0, dot(TT1,TT2), eps);
  EXPECT_NEAR(0, dot(TT2,TT1), eps);

  TT2.setUnit({2,0,1,2});
  EXPECT_NEAR(0, dot(TT1,TT2), eps);
  EXPECT_NEAR(0, dot(TT2,TT1), eps);

  TT1.setUnit({2,0,1,2});
  EXPECT_NEAR(1, dot(TT1,TT2), eps);
  EXPECT_NEAR(1, dot(TT2,TT1), eps);


  TT1.setUnit({0,1,1,2});
  TT2.setUnit({0,1,1,2});
  EXPECT_NEAR(1, dot(TT1,TT2), eps);
  EXPECT_NEAR(1, dot(TT2,TT1), eps);

  TT2.setUnit({1,0,1,2});
  EXPECT_NEAR(0, dot(TT1,TT2), eps);
  EXPECT_NEAR(0, dot(TT2,TT1), eps);
}

TYPED_TEST(PITTS_FixedTensorTrain_dot, rank_4_tensor_random_self)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,3>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT(4), unitTT(4);

  // TT-rank==1
  randomize(TT);
  Type tmp = 0;
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 3; k++)
        for(int l = 0; l < 3; l++)
        {
          unitTT.setUnit({i,j,k,l});
          tmp += std::pow(dot(TT,unitTT),2);
        }
  EXPECT_NEAR(tmp, dot(TT,TT), eps);

  // TT-rank>1
  TT.setTTranks({2,3,2});
  randomize(TT);
  tmp = 0;
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 3; k++)
        for(int l = 0; l < 3; l++)
        {
          unitTT.setUnit({i,j,k,l});
          tmp += std::pow(dot(TT,unitTT),2);
        }
  EXPECT_NEAR(tmp, dot(TT,TT), eps);
}

TYPED_TEST(PITTS_FixedTensorTrain_dot, large_rank_3_tensor_random_other)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,15>;
  constexpr auto eps = 1.e-10;

  FixedTensorTrain TT1(3), TT2(3), unitTT(3);

  // set larger TT-ranks
  TT1.setTTranks({2,5});
  randomize(TT1);
  TT2.setTTranks({3,2});
  randomize(TT2);
  Type tmp = 0;
  for(int i = 0; i < 15; i++)
    for(int j = 0; j < 15; j++)
      for(int k = 0; k < 15; k++)
      {
        unitTT.setUnit({i,j,k});
        tmp += dot(TT1,unitTT) * dot(TT2,unitTT);
      }
  EXPECT_NEAR(tmp, dot(TT1,TT2), eps);
}
