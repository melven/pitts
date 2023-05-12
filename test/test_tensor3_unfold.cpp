#include <gtest/gtest.h>
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3_random.hpp"
#include "pitts_eigen.hpp"


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

TEST(PITTS_Tensor3_unfold, unfold_right_Eigen_scalar)
{
  using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(1, 1, 1);
  t3(0, 0, 0) = 7;

  EigenMatrix mat;
  unfold_right(t3, mat);
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

TEST(PITTS_Tensor3_unfold, unfold_right_Eigen_random)
{
  using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);

  EigenMatrix mat;
  unfold_right(t3, mat);
  ASSERT_EQ(3, mat.rows());
  ASSERT_EQ(5*7, mat.cols());
  for(int i = 0 ; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3(i,j,k), mat(i,j+k*5), eps);
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

TEST(PITTS_Tensor3_unfold, unfold_right_Tensor2_scalar)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(1, 1, 1);
  t3(0, 0, 0) = 7;

  Tensor2_double mat;
  unfold_right(t3, mat);
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

TEST(PITTS_Tensor3_unfold, unfold_right_Tensor2_random)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);

  Tensor2_double mat;
  unfold_right(t3, mat);
  ASSERT_EQ(3, mat.r1());
  ASSERT_EQ(5*7, mat.r2());
  for(int i = 0 ; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3(i,j,k), mat(i,j+k*5), eps);
      }
}

TEST(PITTS_Tensor3_unfold, unfold_vec_Eigen)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  using EigenVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);

  EigenVector vec;
  unfold(t3, vec);
  ASSERT_EQ(3*5*7, vec.size());
  for(int i = 0 ; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3(i,j,k), vec(i+j*3+k*5*3), eps);
      }
}

TEST(PITTS_Tensor3_unfold, unfold_vec_std_vector)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);

  std::vector<double> vec;
  unfold(t3, vec);
  ASSERT_EQ(3*5*7, vec.size());
  for(int i = 0 ; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3(i,j,k), vec.at(i+j*3+k*5*3), eps);
      }
}

TEST(PITTS_Tensor3_unfold, unfold_move_multivector)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);
  Tensor3_double t3_ref;
  copy(t3, t3_ref);

  MultiVector_double mv = unfold(std::move(t3));
  ASSERT_EQ(0, t3.r1());
  ASSERT_EQ(0, t3.n());
  ASSERT_EQ(0, t3.r2());
  ASSERT_EQ(3*5*7, mv.rows());
  ASSERT_EQ(1, mv.cols());
  for(int i = 0 ; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3_ref(i,j,k), mv(i+j*3+k*5*3,0), eps);
      }
}