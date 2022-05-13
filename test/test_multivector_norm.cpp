#include <gtest/gtest.h>
#include "pitts_multivector_norm.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"


TEST(PITTS_MultiVector_norm, single_col)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::Array<double, 1, Eigen::Dynamic>;

  MultiVector_double X(50,1);

  randomize(X);

  const auto result = norm2(X);

  arr result_ref(1);
  result_ref(0) = ConstEigenMap(X).norm();

  ASSERT_NEAR(result_ref, result, eps);
}

TEST(PITTS_MultiVector_norm, simple)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::Array<double, 1, Eigen::Dynamic>;

  MultiVector_double X(2,3);

  arr result_ref(3);

  X(0,0) = 1;
  X(1,0) = 1;
  result_ref(0) = std::sqrt(2.);

  X(0,1) = 3;
  X(1,1) = 4;
  result_ref(1) = 5;

  X(0,2) = -1;
  X(1,2) = 1;
  result_ref(2) = std::sqrt(2.);

  const auto result = norm2(X);


  ASSERT_NEAR(result_ref, result, eps);
}

TEST(PITTS_MultiVector_norm, multi_col)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::Array<double, 1, Eigen::Dynamic>;

  MultiVector_double X(51,5);

  randomize(X);

  const auto result = norm2(X);

  const arr result_ref = (ConstEigenMap(X).transpose() * ConstEigenMap(X)).diagonal().array().sqrt();

  ASSERT_NEAR(result_ref, result, eps);
}

