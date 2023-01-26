#include <gtest/gtest.h>
#include "pitts_tensortrain_sweep_index.hpp"
#include <array>

namespace
{
  using PITTS::internal::SweepIndex;
}

TEST(PITTS_TensorTrain_SweepIndex, constructor)
{
  SweepIndex swpIdx(10, 2, 1);

  EXPECT_EQ(10, swpIdx.nDim());
  EXPECT_EQ(2, swpIdx.nMALS());
  EXPECT_EQ(1, swpIdx.nOverlap());
  EXPECT_EQ(0, swpIdx.leftDim());
  EXPECT_EQ(1, swpIdx.rightDim());
}

TEST(PITTS_TensorTrain_SweepIndex, dimension_checks)
{
  EXPECT_THROW(SweepIndex(10, 11, 0), std::invalid_argument);
  EXPECT_THROW(SweepIndex(10, 0, 0), std::invalid_argument);
  EXPECT_THROW(SweepIndex(10, 4, 4), std::invalid_argument);

  EXPECT_NO_THROW(SweepIndex(5, 1, 0));
  EXPECT_NO_THROW(SweepIndex(5, 2, 0));
  EXPECT_NO_THROW(SweepIndex(5, 2, 1));
  EXPECT_NO_THROW(SweepIndex(5, 3, 0));
  EXPECT_NO_THROW(SweepIndex(5, 3, 1));
  EXPECT_NO_THROW(SweepIndex(5, 3, 2));
  EXPECT_NO_THROW(SweepIndex(5, 5, 1));
  EXPECT_NO_THROW(SweepIndex(5, 5, 4));
}

TEST(PITTS_TensorTrain_SweepIndex, index_checks)
{
  EXPECT_FALSE(SweepIndex(5, 1, 0, -1));
  EXPECT_TRUE(SweepIndex(5, 1, 0, 0));
  EXPECT_TRUE(SweepIndex(5, 1, 0, 1));
  EXPECT_TRUE(SweepIndex(5, 1, 0, 2));
  EXPECT_TRUE(SweepIndex(5, 1, 0, 3));
  EXPECT_TRUE(SweepIndex(5, 1, 0, 4));
  EXPECT_FALSE(SweepIndex(5, 1, 0, 5));

  EXPECT_FALSE(SweepIndex(5, 2, 1, -1));
  EXPECT_TRUE(SweepIndex(5, 2, 1, 0));
  EXPECT_TRUE(SweepIndex(5, 2, 1, 1));
  EXPECT_TRUE(SweepIndex(5, 2, 1, 2));
  EXPECT_TRUE(SweepIndex(5, 2, 1, 3));
  EXPECT_FALSE(SweepIndex(5, 2, 1, 4));
}

TEST(PITTS_TensorTrain_SweepIndex, first_and_last)
{
  {
    SweepIndex swpIdx(5, 1, 0, 2);
    EXPECT_EQ(SweepIndex(5,1,0,0), swpIdx.first());
    EXPECT_EQ(SweepIndex(5,1,0,4), swpIdx.last());
  }

  {
    SweepIndex swpIdx(5, 2, 1, 2);
    EXPECT_EQ(SweepIndex(5,2,1,0), swpIdx.first());
    EXPECT_EQ(SweepIndex(5,2,1,3), swpIdx.last());
  }

  {
    SweepIndex swpIdx(7, 5, 0, 2);
    EXPECT_EQ(SweepIndex(7,5,0,0), swpIdx.first());
    EXPECT_EQ(SweepIndex(7,5,0,2), swpIdx.last());
  }
}

TEST(PITTS_TensorTrain_SweepIndex, sweep_left_to_right_ALS)
{
  const std::array expectedLeftDim = {0, 1, 2, 3, 4};
  const std::array expectedRightDim = {0, 1, 2, 3, 4};

  int iter = 0;
  for(auto swpIdx = SweepIndex(5,1,0).first(); swpIdx; swpIdx = swpIdx.next())
  {
    EXPECT_EQ(expectedLeftDim.at(iter), swpIdx.leftDim());
    EXPECT_EQ(expectedRightDim.at(iter), swpIdx.rightDim());

    iter++;
  }
  ASSERT_EQ(expectedLeftDim.size(), iter);
}

TEST(PITTS_TensorTrain_SweepIndex, sweep_right_to_left_ALS)
{
  const std::array expectedLeftDim = {4, 3, 2, 1, 0};
  const std::array expectedRightDim = {4, 3, 2, 1, 0};

  int iter = 0;
  for(auto swpIdx = SweepIndex(5,1,0).last(); swpIdx; swpIdx = swpIdx.previous())
  {
    EXPECT_EQ(expectedLeftDim.at(iter), swpIdx.leftDim());
    EXPECT_EQ(expectedRightDim.at(iter), swpIdx.rightDim());

    iter++;
  }
  ASSERT_EQ(expectedLeftDim.size(), iter);
}

TEST(PITTS_TensorTrain_SweepIndex, sweep_left_to_right_MALS)
{
  const std::array expectedLeftDim = {0, 1, 2, 3};
  const std::array expectedRightDim = {1, 2, 3, 4};

  int iter = 0;
  for(auto swpIdx = SweepIndex(5,2,1).first(); swpIdx; swpIdx = swpIdx.next())
  {
    EXPECT_EQ(expectedLeftDim.at(iter), swpIdx.leftDim());
    EXPECT_EQ(expectedRightDim.at(iter), swpIdx.rightDim());

    iter++;
  }
  ASSERT_EQ(expectedLeftDim.size(), iter);
}

TEST(PITTS_TensorTrain_SweepIndex, sweep_right_to_left_MALS)
{
  const std::array expectedLeftDim = {3, 2, 1, 0};
  const std::array expectedRightDim = {4, 3, 2, 1};

  int iter = 0;
  for(auto swpIdx = SweepIndex(5,2,1).last(); swpIdx; swpIdx = swpIdx.previous())
  {
    EXPECT_EQ(expectedLeftDim.at(iter), swpIdx.leftDim());
    EXPECT_EQ(expectedRightDim.at(iter), swpIdx.rightDim());

    iter++;
  }
  ASSERT_EQ(expectedLeftDim.size(), iter);
}

TEST(PITTS_TensorTrain_SweepIndex, sweep_left_to_right_ttGMRES)
{
  const std::array expectedLeftDim = {0};
  const std::array expectedRightDim = {4};

  int iter = 0;
  for(auto swpIdx = SweepIndex(5,5,0).first(); swpIdx; swpIdx = swpIdx.next())
  {
    EXPECT_EQ(expectedLeftDim.at(iter), swpIdx.leftDim());
    EXPECT_EQ(expectedRightDim.at(iter), swpIdx.rightDim());

    iter++;
  }
  ASSERT_EQ(expectedLeftDim.size(), iter);
}

TEST(PITTS_TensorTrain_SweepIndex, sweep_right_to_left_ttGMRES)
{
  const std::array expectedLeftDim = {0};
  const std::array expectedRightDim = {4};

  int iter = 0;
  for(auto swpIdx = SweepIndex(5,5,0).last(); swpIdx; swpIdx = swpIdx.previous())
  {
    EXPECT_EQ(expectedLeftDim.at(iter), swpIdx.leftDim());
    EXPECT_EQ(expectedRightDim.at(iter), swpIdx.rightDim());

    iter++;
  }
  ASSERT_EQ(expectedLeftDim.size(), iter);
}

TEST(PITTS_TensorTrain_SweepIndex, sweep_left_to_right_MALS_13_4_2)
{
  const std::array expectedLeftDim = {0, 2, 4, 6, 8, 9};
  const std::array expectedRightDim = {3, 5, 7, 9, 11, 12};

  int iter = 0;
  for(auto swpIdx = SweepIndex(13,4,2).first(); swpIdx; swpIdx = swpIdx.next())
  {
    EXPECT_EQ(expectedLeftDim.at(iter), swpIdx.leftDim());
    EXPECT_EQ(expectedRightDim.at(iter), swpIdx.rightDim());

    iter++;
  }
  ASSERT_EQ(expectedLeftDim.size(), iter);
}

TEST(PITTS_TensorTrain_SweepIndex, sweep_right_to_left_MALS_13_4_2)
{
  const std::array expectedLeftDim = {9, 7, 5, 3, 1, 0};
  const std::array expectedRightDim = {12, 10, 8, 6, 4, 3};

  int iter = 0;
  for(auto swpIdx = SweepIndex(13,4,2).last(); swpIdx; swpIdx = swpIdx.previous())
  {
    EXPECT_EQ(expectedLeftDim.at(iter), swpIdx.leftDim());
    EXPECT_EQ(expectedRightDim.at(iter), swpIdx.rightDim());

    iter++;
  }
  ASSERT_EQ(expectedLeftDim.size(), iter);
}