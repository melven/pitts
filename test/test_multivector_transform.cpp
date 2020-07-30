#include <gtest/gtest.h>
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include <Eigen/Dense>
#include "eigen_test_helper.hpp"


TEST(PITTS_MultiVector_transform, invalid_args)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(50,3), Y;
  Tensor2_double M(1,1);

  EXPECT_THROW(transform(X, M, Y), std::invalid_argument);

  M.resize(3,2);

  EXPECT_NO_THROW(transform(X, M, Y));

  EXPECT_THROW(transform(X, M, Y, {10,2}), std::invalid_argument);

  EXPECT_NO_THROW(transform(X, M, Y, {10,10}));
}

TEST(PITTS_MultiVector_transform, single_col_no_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(50,1), Y;
  Tensor2_double M(1,1);

  randomize(X);
  randomize(M);

  transform(X, M, Y);

  const Eigen::MatrixXd Y_ref = ConstEigenMap(X) * ConstEigenMap(M);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, single_col_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(100,1), Y;
  Tensor2_double M(1,1);

  randomize(X);
  randomize(M);

  transform(X, M, Y, {50, 2});

  ASSERT_EQ(50, Y.rows());
  ASSERT_EQ(2, Y.cols());

  const Eigen::MatrixXd Ytmp = ConstEigenMap(X) * ConstEigenMap(M);
  const Eigen::MatrixXd Y_ref = Eigen::Map<const Eigen::MatrixXd>(Ytmp.data(), 50, 2);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, multi_col_no_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(50,3), Y;
  Tensor2_double M(3,2);

  randomize(X);
  randomize(M);

  transform(X, M, Y);

  const Eigen::MatrixXd Y_ref = ConstEigenMap(X) * ConstEigenMap(M);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, multi_col_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(100,3), Y;
  Tensor2_double M(3,2);

  randomize(X);
  randomize(M);

  transform(X, M, Y, {50, 4});

  ASSERT_EQ(50, Y.rows());
  ASSERT_EQ(4, Y.cols());

  const Eigen::MatrixXd Ytmp = ConstEigenMap(X) * ConstEigenMap(M);
  const Eigen::MatrixXd Y_ref = Eigen::Map<const Eigen::MatrixXd>(Ytmp.data(), 50, 4);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, large_no_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(117,4), Y;
  Tensor2_double M(4,5);

  randomize(X);
  randomize(M);

  transform(X, M, Y);

  const Eigen::MatrixXd Y_ref = ConstEigenMap(X) * ConstEigenMap(M);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, large_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(117,4), Y;
  Tensor2_double M(4,5);

  randomize(X);
  randomize(M);

  transform(X, M, Y, {195, 3});

  ASSERT_EQ(195, Y.rows());
  ASSERT_EQ(3, Y.cols());

  const Eigen::MatrixXd Ytmp = ConstEigenMap(X) * ConstEigenMap(M);
  const Eigen::MatrixXd Y_ref = Eigen::Map<const Eigen::MatrixXd>(Ytmp.data(), 195, 3);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, large_reshape2)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(117,4), Y;
  Tensor2_double M(4,5);

  randomize(X);
  randomize(M);

  transform(X, M, Y, {39, 15});

  ASSERT_EQ(39, Y.rows());
  ASSERT_EQ(15, Y.cols());

  const Eigen::MatrixXd Ytmp = ConstEigenMap(X) * ConstEigenMap(M);
  const Eigen::MatrixXd Y_ref = Eigen::Map<const Eigen::MatrixXd>(Ytmp.data(), 39, 15);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}
