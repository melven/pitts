#include <gtest/gtest.h>
#include "pitts_tensortrain_dot.hpp"

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
