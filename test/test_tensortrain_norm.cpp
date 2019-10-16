#include <gtest/gtest.h>
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_random.hpp"

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

  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);
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

  // TT-ranks==1
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);

  // higher rank
  TT.setTTranks({3});
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);
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

  // TT-ranks==1
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);

  // higher TT-ranks
  TT.setTTranks({3,1,2});
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);
}

TEST(PITTS_TensorTrain_norm, large_rank_4_tensor)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(4,100);

  TT.setZero();
  EXPECT_EQ(0, norm2(TT));

  TT.setOnes();
  EXPECT_NEAR(std::sqrt(100*100*100*100), norm2(TT), eps);

  TT.setUnit({0,1,1,0});
  EXPECT_NEAR(1, norm2(TT), eps);

  // TT-ranks==1
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);

  // higher TT-ranks
  TT.setTTranks({3,1,2});
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);
}
