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

TEST(PITTS_MultiVector_dot, single_col)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::Array<double, Eigen::Dynamic, 1>;

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
  using arr = Eigen::Array<double, Eigen::Dynamic, 1>;

  MultiVector_double X(51,5), Y(51,5);

  randomize(X);
  randomize(Y);

  const auto result = dot(X, Y);

  const arr result_ref = (ConstEigenMap(X).transpose() * ConstEigenMap(Y)).diagonal();

  ASSERT_NEAR(result_ref, result, eps);
}

