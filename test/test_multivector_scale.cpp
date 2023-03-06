#include <gtest/gtest.h>
#include "pitts_multivector_scale.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"


TEST(PITTS_MultiVector_scale, single_col)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(50,1), X_ref(50,1);

  randomize(X);

  const arr alpha = arr::Constant(1, 37.5);

  EigenMap(X_ref) = ConstEigenMap(X) * alpha(0);

  scale(alpha, X);

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}

TEST(PITTS_MultiVector_scale, simple)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(2,3), X_ref(2,3);

  arr alpha(3);

  X(0,0) = 1;
  X(1,0) = 1;
  alpha(0) = 2;
  X_ref(0,0) = 2;
  X_ref(1,0) = 2;

  X(0,1) = 3;
  X(1,1) = 4;
  alpha(1) = -3;
  X_ref(0,1) = -9;
  X_ref(1,1) = -12;

  X(0,2) = -1;
  X(1,2) = 1;
  alpha(2) = 7.5;
  X_ref(0,2) = -7.5;
  X_ref(1,2) = 7.5;

  scale(alpha, X);

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}

TEST(PITTS_MultiVector_scale, multi_col)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(51,5), X_ref(51,5);

  randomize(X);

  arr alpha(5);
  alpha << 37.4,2.,3.,42.,-1.;

  EigenMap(X_ref) = ConstEigenMap(X) * alpha.matrix().asDiagonal();

  scale(alpha, X);

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}

