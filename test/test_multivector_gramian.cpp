#include <gtest/gtest.h>
#include "pitts_multivector_gramian.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"


TEST(PITTS_MultiVector_gramian, simple)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixXd;

  MultiVector_double X(2,3);

  EigenMap(X) << 1, 2, 3,
                 4, 5, 6;

  Tensor2_double result;
  gramian(X, result);

  const mat result_ref = ConstEigenMap(X).transpose() * ConstEigenMap(X);
  ASSERT_NEAR(result_ref, ConstEigenMap(result), eps);
}

TEST(PITTS_MultiVector_gramian, single_col)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixXd;

  MultiVector_double X(50, 1);
  randomize(X);

  Tensor2_double result;
  gramian(X, result);

  const mat result_ref = ConstEigenMap(X).transpose() * ConstEigenMap(X);
  ASSERT_NEAR(result_ref, ConstEigenMap(result), eps);
}

TEST(PITTS_MultiVector_gramian, multi_col)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixXd;

  MultiVector_double X(50, 5);
  randomize(X);

  Tensor2_double result;
  gramian(X, result);

  const mat result_ref = ConstEigenMap(X).transpose() * ConstEigenMap(X);
  ASSERT_NEAR(result_ref, ConstEigenMap(result), eps);
}

