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
