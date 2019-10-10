#include <gtest/gtest.h>
#include "pitts_tensortrain_norm.hpp"

TEST(PITTS_TensorTrain_norm, rank_1_vector)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(1,5);

  TT.setZero();
  EXPECT_EQ(0, norm2(TT));

  TT.setOnes();
  EXPECT_NEAR(std::sqrt(5.), norm2(TT), eps);

  TT.setUnit({2});
  EXPECT_NEAR(1, norm2(TT), eps);
}

TEST(PITTS_TensorTrain_norm, rank_2_matrix)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(2,5);

  TT.setZero();
  EXPECT_EQ(0, norm2(TT));

  TT.setOnes();
  EXPECT_NEAR(std::sqrt(25.), norm2(TT), eps);

  TT.setUnit({2,1});
  EXPECT_NEAR(1, norm2(TT), eps);
}

TEST(PITTS_TensorTrain_norm, rank_4_tensor)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(4,2);

  TT.setZero();
  EXPECT_EQ(0, norm2(TT));

  TT.setOnes();
  EXPECT_NEAR(std::sqrt(2*2*2*2), norm2(TT), eps);

  TT.setUnit({0,1,1,0});
  EXPECT_NEAR(1, norm2(TT), eps);
}
