#include <gtest/gtest.h>
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3_random.hpp"
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options


TEST(PITTS_Tensor3_unfold, unfold_left_Eigen_scalar)
{
  using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(1, 1, 1);
  t3(0, 0, 0) = 7;

  EigenMatrix mat;
  unfold_left(t3, mat);
  ASSERT_EQ(1, mat.rows());
  ASSERT_EQ(1, mat.cols());
  EXPECT_NEAR(7., mat(0, 0), eps);
}

TEST(PITTS_Tensor3_unfold, unfold_left_Eigen_random)
{
  using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);

  EigenMatrix mat;
  unfold_left(t3, mat);
  ASSERT_EQ(3*5, mat.rows());
  ASSERT_EQ(7, mat.cols());
  for(int i = 0 ; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3(i,j,k), mat(i+j*3,k), eps);
      }
}

TEST(PITTS_Tensor3_unfold, unfold_left_Tensor2_scalar)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(1, 1, 1);
  t3(0, 0, 0) = 7;

  Tensor2_double mat;
  unfold_left(t3, mat);
  ASSERT_EQ(1, mat.r1());
  ASSERT_EQ(1, mat.r2());
  EXPECT_NEAR(7., mat(0, 0), eps);
}

TEST(PITTS_Tensor3_unfold, unfold_left_Tensor2_random)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);

  Tensor2_double mat;
  unfold_left(t3, mat);
  ASSERT_EQ(3*5, mat.r1());
  ASSERT_EQ(7, mat.r2());
  for(int i = 0 ; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3(i,j,k), mat(i+j*3,k), eps);
      }
}
