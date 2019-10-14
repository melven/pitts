#include <gtest/gtest.h>
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_random.hpp"

TEST(PITTS_TensorTrain_dot, rank_1_vector_self)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(1,5);

  TT.setZero();
  EXPECT_EQ(0, dot(TT,TT));

  TT.setOnes();
  EXPECT_NEAR(5., dot(TT,TT), eps);

  TT.setUnit({2});
  EXPECT_NEAR(1, dot(TT,TT), eps);
}


TEST(PITTS_TensorTrain_dot, large_rank_1_vector_self)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(1,50);

  TT.setZero();
  EXPECT_EQ(0, dot(TT,TT));

  TT.setOnes();
  EXPECT_NEAR(50., dot(TT,TT), eps);

  TT.setUnit({2});
  EXPECT_NEAR(1, dot(TT,TT), eps);
}


TEST(PITTS_TensorTrain_dot, rank_1_vector_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(1,5), unitTT(1,5);

  randomize(TT);
  double tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    unitTT.setUnit({i});
    tmp += std::pow(dot(TT,unitTT),2);
  }
  EXPECT_NEAR(tmp, dot(TT,TT), eps);
}

TEST(PITTS_TensorTrain_dot, rank_1_vector_random_other)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(1,5), TT2(1,5), unitTT(1,5);

  randomize(TT1);
  randomize(TT2);
  double tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    unitTT.setUnit({i});
    tmp += dot(TT1,unitTT) * dot(TT2,unitTT);
  }
  EXPECT_NEAR(tmp, dot(TT1,TT2), eps);
}

TEST(PITTS_TensorTrain_dot, rank_2_matrix_self)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(2,5);

  TT.setZero();
  EXPECT_EQ(0, dot(TT,TT));

  TT.setOnes();
  EXPECT_NEAR(25., dot(TT,TT), eps);

  TT.setUnit({2,1});
  EXPECT_NEAR(1, dot(TT,TT), eps);
}

TEST(PITTS_TensorTrain_dot, rank_2_matrix_random_self)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(2,5), unitTT(2,5);

  // TT-rank==1
  randomize(TT);
  double tmp = 0;
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
  TT.setTTranks({3});
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

TEST(PITTS_TensorTrain_dot, rank_2_matrix_random_other)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(2,5), TT2(2,5), unitTT(2,5);

  // TT-rank==1
  randomize(TT1);
  randomize(TT2);
  double tmp = 0;
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
  TT1.setTTranks({3});
  randomize(TT1);
  TT2.setTTranks({2});
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


TEST(PITTS_TensorTrain_dot, rank_1_vector_unit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(1,3), TT2(1,3);

  TT1.setUnit({0});
  TT2.setUnit({0});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));

  TT2.setUnit({1});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT2.setUnit({2});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT1.setUnit({2});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));
}

TEST(PITTS_TensorTrain_dot, rank_2_matrix_unit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(2,3), TT2(2,3);

  TT1.setUnit({0,0});
  TT2.setUnit({0,0});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));

  TT2.setUnit({1,0});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT2.setUnit({2,0});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT1.setUnit({2,0});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));


  TT1.setUnit({0,1});
  TT2.setUnit({0,1});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));

  TT2.setUnit({1,0});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));
}

TEST(PITTS_TensorTrain_dot, rank_4_tensor_unit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(4,3), TT2(4,3);

  TT1.setUnit({0,0,1,2});
  TT2.setUnit({0,0,1,2});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));

  TT2.setUnit({1,0,1,2});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT2.setUnit({2,0,1,2});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT1.setUnit({2,0,1,2});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));


  TT1.setUnit({0,1,1,2});
  TT2.setUnit({0,1,1,2});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));

  TT2.setUnit({1,0,1,2});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));
}

TEST(PITTS_TensorTrain_dot, rank_4_tensor_random_self)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(4,3), unitTT(4,3);

  // TT-rank==1
  randomize(TT);
  double tmp = 0;
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

TEST(PITTS_TensorTrain_dot, large_rank_3_tensor_random_other)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(3,15), TT2(3,15), unitTT(3,15);

  // set larger TT-ranks
  TT1.setTTranks({2,5});
  randomize(TT1);
  TT2.setTTranks({3,2});
  randomize(TT2);
  double tmp = 0;
  for(int i = 0; i < 15; i++)
    for(int j = 0; j < 15; j++)
      for(int k = 0; k < 15; k++)
      {
        unitTT.setUnit({i,j,k});
        tmp += dot(TT1,unitTT) * dot(TT2,unitTT);
      }
  EXPECT_NEAR(tmp, dot(TT1,TT2), eps);
}
