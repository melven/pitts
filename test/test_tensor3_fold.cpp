#include <gtest/gtest.h>
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3_random.hpp"
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options


// anonymous namespace
namespace
{
  template<typename T>
  void check_equal(const PITTS::Tensor3<T>& t3_ref, const PITTS::Tensor3<T>& t3, T eps)
  {
    ASSERT_EQ(t3_ref.r1(), t3.r1());
    ASSERT_EQ(t3_ref.n(), t3.n());
    ASSERT_EQ(t3_ref.r2(), t3.r2());

    for(int i = 0; i < t3_ref.r1(); i++)
      for(int j = 0; j < t3_ref.n(); j++)
        for(int k = 0; k < t3_ref.r2(); k++)
        {
          EXPECT_NEAR(t3_ref(i,j,k), t3(i,j,k), eps);
        }
  }
}

TEST(PITTS_Tensor3_fold, fold_left_Eigen_scalar)
{
  using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  EigenMatrix mat(1,1);
  mat(0,0) = 23;

  Tensor3_double t3;

  fold_left(mat, 1, t3);

  ASSERT_EQ(1, t3.r1());
  ASSERT_EQ(1, t3.n());
  ASSERT_EQ(1, t3.r2());
  EXPECT_NEAR(23., t3(0,0,0), eps);
}

TEST(PITTS_Tensor3_fold, fold_right_Eigen_scalar)
{
  using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  EigenMatrix mat(1,1);
  mat(0,0) = 23;

  Tensor3_double t3;

  fold_right(mat, 1, t3);

  ASSERT_EQ(1, t3.r1());
  ASSERT_EQ(1, t3.n());
  ASSERT_EQ(1, t3.r2());
  EXPECT_NEAR(23., t3(0,0,0), eps);
}

TEST(PITTS_Tensor3_fold, unfold_fold_left_Eigen_random)
{
  using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3_ref(3, 5, 7);
  randomize(t3_ref);

  EigenMatrix mat;
  unfold_left(t3_ref, mat);

  Tensor3_double t3;
  fold_left(mat, 5, t3);

  check_equal(t3_ref, t3, eps);
}

TEST(PITTS_Tensor3_fold, unfold_fold_right_Eigen_random)
{
  using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3_ref(3, 5, 7);
  randomize(t3_ref);

  EigenMatrix mat;
  unfold_right(t3_ref, mat);

  Tensor3_double t3;
  fold_right(mat, 5, t3);

  check_equal(t3_ref, t3, eps);
}

TEST(PITTS_Tensor3_fold, fold_left_Tensor2_scalar)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor2_double mat(1, 1);
  mat(0, 0) = 23;

  Tensor3_double t3;

  fold_left(mat, 1, t3);

  ASSERT_EQ(1, t3.r1());
  ASSERT_EQ(1, t3.n());
  ASSERT_EQ(1, t3.r2());
  EXPECT_NEAR(23., t3(0, 0, 0), eps);
}

TEST(PITTS_Tensor3_fold, fold_right_Tensor2_scalar)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor2_double mat(1, 1);
  mat(0, 0) = 23;

  Tensor3_double t3;

  fold_right(mat, 1, t3);

  ASSERT_EQ(1, t3.r1());
  ASSERT_EQ(1, t3.n());
  ASSERT_EQ(1, t3.r2());
  EXPECT_NEAR(23., t3(0, 0, 0), eps);
}

TEST(PITTS_Tensor3_fold, unfold_fold_left_Tensor2_random)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3_ref(3, 5, 7);
  randomize(t3_ref);

  Tensor2_double mat;
  unfold_left(t3_ref, mat);

  Tensor3_double t3;
  fold_left(mat, 5, t3);

  check_equal(t3_ref, t3, eps);
}

TEST(PITTS_Tensor3_fold, unfold_fold_right_Tensor2_random)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3_ref(3, 5, 7);
  randomize(t3_ref);

  Tensor2_double mat;
  unfold_left(t3_ref, mat);

  Tensor3_double t3;
  fold_left(mat, 5, t3);

  check_equal(t3_ref, t3, eps);
}

TEST(PITTS_Tensor3_fold, fold_vec_Eigen)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  using EigenVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
  constexpr auto eps = 1.e-10;


  Tensor3_double t3_ref(3, 5, 7);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 7; k++)
        t3_ref(i,j,k) = i+10*j+100*k;

  EigenVector v(3*5*7);
  for(int i = 0; i < 3*5*7; i++)
    v(i) = t3_ref(i % 3, (i/3)%5, (i/3)/5);
  
  Tensor3_double t3;
  fold(v, 3, 5, 7, t3);

  check_equal(t3_ref, t3, eps);
}

TEST(PITTS_Tensor3_fold, fold_vec_std_vector)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;


  Tensor3_double t3_ref(3, 5, 7);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 7; k++)
        t3_ref(i,j,k) = i+10*j+100*k;

  std::vector<double> v(3*5*7);
  for(int i = 0; i < 3*5*7; i++)
    v[i] = t3_ref(i % 3, (i/3)%5, (i/3)/5);
  
  Tensor3_double t3;
  fold(v, 3, 5, 7, t3);

  check_equal(t3_ref, t3, eps);
}