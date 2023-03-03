#include <gtest/gtest.h>
#include "pitts_multivector_dot.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"


TEST(PITTS_MultiVector_dot, invalid_args)
{
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(50,3), Y(49,3);

  EXPECT_THROW(dot(X, Y), std::invalid_argument);

  Y.resize(50,3);

  EXPECT_NO_THROW(dot(X, Y));

  X.resize(50,1);

  EXPECT_THROW(dot(X, Y), std::invalid_argument);
}

TEST(PITTS_MultiVector_dot, simple)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(2,3), Y(2,3);

  arr result_ref(3);

  X(0,0) = 1; Y(0,0) = 0;
  X(1,0) = 1; Y(1,0) = 1;
  result_ref(0) = 1;

  X(0,1) = 2; Y(0,1) = 1;
  X(1,1) = 1; Y(1,1) = 2;
  result_ref(1) = 4;

  X(0,2) = 3; Y(0,2) = 3;
  X(1,2) = 4; Y(1,2) = 4;
  result_ref(2) = 25;

  const auto result = dot(X, Y);

  ASSERT_NEAR(result_ref, result, eps);
}

TEST(PITTS_MultiVector_dot, single_col)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(50,1), Y(50,1);

  randomize(X);
  randomize(Y);

  const auto result = dot(X, Y);

  const arr result_ref = ConstEigenMap(X).transpose() * ConstEigenMap(Y);

  ASSERT_NEAR(result_ref, result, eps);
}

TEST(PITTS_MultiVector_dot, multi_col)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(51,5), Y(51,5);

  randomize(X);
  randomize(Y);

  const auto result = dot(X, Y);

  const arr result_ref = (ConstEigenMap(X).transpose() * ConstEigenMap(Y)).diagonal();

  ASSERT_NEAR(result_ref, result, eps);
}

