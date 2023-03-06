#include <gtest/gtest.h>
#include "pitts_multivector_axpby.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"


TEST(PITTS_MultiVector_axpby, axpy_invalid_args)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(50,3), Y(49,3);
  arr alpha(3);

  EXPECT_THROW(axpy(alpha, X, Y), std::invalid_argument);

  Y.resize(50,3);

  EXPECT_NO_THROW(axpy(alpha, X, Y));

  X.resize(50,1);

  EXPECT_THROW(axpy(alpha, X, Y), std::invalid_argument);

  Y.resize(50,1);
  
  EXPECT_THROW(axpy(alpha, X, Y), std::invalid_argument);
}

TEST(PITTS_MultiVector_axpby, axpy_random)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(51,5), Y(51,5), Y_ref(51,5);
  randomize(X);
  randomize(Y);

  arr alpha = arr::Random(5);

  EigenMap(Y_ref) = ConstEigenMap(Y) + ConstEigenMap(X) * alpha.matrix().asDiagonal();

  axpy(alpha, X, Y);

  ASSERT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);
}


TEST(PITTS_MultiVector_axpby, axpy_norm2_invalid_args)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(50,3), Y(49,3);
  arr alpha(3);

  EXPECT_THROW(axpy_norm2(alpha, X, Y), std::invalid_argument);

  Y.resize(50,3);

  EXPECT_NO_THROW(axpy_norm2(alpha, X, Y));

  X.resize(50,1);

  EXPECT_THROW(axpy_norm2(alpha, X, Y), std::invalid_argument);

  Y.resize(50,1);
  
  EXPECT_THROW(axpy_norm2(alpha, X, Y), std::invalid_argument);
}

TEST(PITTS_MultiVector_axpby, axpy_norm2_random)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(51,5), Y(51,5), Y_ref(51,5);
  randomize(X);
  randomize(Y);

  arr alpha = arr::Random(5);

  EigenMap(Y_ref) = ConstEigenMap(Y) + ConstEigenMap(X) * alpha.matrix().asDiagonal();

  const arr nrm = axpy_norm2(alpha, X, Y);

  const arr nrm_ref = (ConstEigenMap(Y_ref).transpose() * ConstEigenMap(Y_ref)).diagonal().array().sqrt();

  ASSERT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);
  ASSERT_NEAR(nrm_ref, nrm, eps);
}


TEST(PITTS_MultiVector_axpby, axpy_dot_invalid_args)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(50,3), Y(49,3), Z(50,3);
  arr alpha(3);

  EXPECT_THROW(axpy_dot(alpha, X, Y, Z), std::invalid_argument);

  Y.resize(50,3);

  EXPECT_NO_THROW(axpy_dot(alpha, X, Y, Z));

  X.resize(50,1);

  EXPECT_THROW(axpy_dot(alpha, X, Y, Z), std::invalid_argument);

  Y.resize(50,1);
  
  EXPECT_THROW(axpy_dot(alpha, X, Y, Z), std::invalid_argument);
}

TEST(PITTS_MultiVector_axpby, axpy_dot_random)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;

  MultiVector_double X(51,5), Y(51,5), Y_ref(51,5), Z(51,5);
  randomize(X);
  randomize(Y);
  randomize(Z);

  arr alpha = arr::Random(5);

  EigenMap(Y_ref) = ConstEigenMap(Y) + ConstEigenMap(X) * alpha.matrix().asDiagonal();

  const arr result = axpy_dot(alpha, X, Y, Z);

  const arr result_ref = (ConstEigenMap(Y_ref).transpose() * ConstEigenMap(Z)).diagonal();

  ASSERT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);
  ASSERT_NEAR(result_ref, result, eps);
}

