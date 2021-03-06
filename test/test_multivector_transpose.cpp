#include <gtest/gtest.h>
#include "pitts_multivector.hpp"
#include "pitts_multivector_transpose.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include <Eigen/Dense>
#include "eigen_test_helper.hpp"


TEST(PITTS_MultiVector_transpose, invalid_args)
{
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(50,3), Y;

  EXPECT_THROW(transpose(X, Y, {100,1}), std::invalid_argument);

  EXPECT_NO_THROW(transpose(X, Y, {50,3}));
  EXPECT_NO_THROW(transpose(X, Y, {150,1}));
}

TEST(PITTS_MultiVector_transpose, small_example_no_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(5,2), Y;
  X(0,0) = 1; X(0,1) = 6;
  X(1,0) = 2; X(1,1) = 7;
  X(2,0) = 3; X(2,1) = 8;
  X(3,0) = 4; X(3,1) = 9;
  X(4,0) = 5; X(4,1) = 10;

  transpose(X, Y);

  ASSERT_EQ(2, Y.rows());
  ASSERT_EQ(5, Y.cols());
  MultiVector_double Y_ref(2, 5);
  Y_ref(0,0) = 1; Y_ref(0,1) = 2; Y_ref(0,2) = 3; Y_ref(0,3) = 4; Y_ref(0,4) = 5;
  Y_ref(1,0) = 6; Y_ref(1,1) = 7; Y_ref(1,2) = 8; Y_ref(1,3) = 9; Y_ref(1,4) = 10;
  ASSERT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, small_example_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(5,2), Y;
  X(0,0) = 1; X(0,1) = 6;
  X(1,0) = 2; X(1,1) = 7;
  X(2,0) = 3; X(2,1) = 8;
  X(3,0) = 4; X(3,1) = 9;
  X(4,0) = 5; X(4,1) = 10;

  transpose(X, Y, {5,2});

  ASSERT_EQ(5, Y.rows());
  ASSERT_EQ(2, Y.cols());
  MultiVector_double Y_ref(5, 2);
  Y_ref(0,0) = 1; Y_ref(0,1) = 2;
  Y_ref(1,0) = 3; Y_ref(1,1) = 4;
  Y_ref(2,0) = 5; Y_ref(2,1) = 6;
  Y_ref(3,0) = 7; Y_ref(3,1) = 8;
  Y_ref(4,0) = 9; Y_ref(4,1) = 10;
  ASSERT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, small_example2_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(6,2), Y;
  X(0,0) = 1; X(0,1) = 7;
  X(1,0) = 2; X(1,1) = 8;
  X(2,0) = 3; X(2,1) = 9;
  X(3,0) = 4; X(3,1) = 10;
  X(4,0) = 5; X(4,1) = 11;
  X(5,0) = 6; X(5,1) = 12;

  transpose(X, Y, {4,3});

  ASSERT_EQ(4, Y.rows());
  ASSERT_EQ(3, Y.cols());
  MultiVector_double Y_ref(4, 3);
  Y_ref(0,0) = 1;  Y_ref(0,1) = 2;  Y_ref(0,2) = 3;
  Y_ref(1,0) = 4;  Y_ref(1,1) = 5;  Y_ref(1,2) = 6;
  Y_ref(2,0) = 7;  Y_ref(2,1) = 8;  Y_ref(2,2) = 9;
  Y_ref(3,0) = 10; Y_ref(3,1) = 11; Y_ref(3,2) = 12;
  ASSERT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, small_example2_reshape_reverse)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(6,2), Y;
  X(0,0) = 1;  X(0,1) = 2;
  X(1,0) = 3;  X(1,1) = 4;
  X(2,0) = 5;  X(2,1) = 6;
  X(3,0) = 7;  X(3,1) = 8;
  X(4,0) = 9;  X(4,1) = 10;
  X(5,0) = 11; X(5,1) = 12;

  transpose(X, Y, {4,3}, true);

  ASSERT_EQ(4, Y.rows());
  ASSERT_EQ(3, Y.cols());
  MultiVector_double Y_ref(4, 3);
  Y_ref(0,0) = 1; Y_ref(0,1) = 5; Y_ref(0,2) = 9;
  Y_ref(1,0) = 2; Y_ref(1,1) = 6; Y_ref(1,2) = 10;
  Y_ref(2,0) = 3; Y_ref(2,1) = 7; Y_ref(2,2) = 11;
  Y_ref(3,0) = 4; Y_ref(3,1) = 8; Y_ref(3,2) = 12;
  ASSERT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, small_example2_reshape_reverse2)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(4,3), Y;
  X(0,0) = 1;  X(0,1) = 2;  X(0,2) = 3;
  X(1,0) = 4;  X(1,1) = 5;  X(1,2) = 6;
  X(2,0) = 7;  X(2,1) = 8;  X(2,2) = 9;
  X(3,0) = 10; X(3,1) = 11; X(3,2) = 12;

  transpose(X, Y, {6,2}, true);

  ASSERT_EQ(6, Y.rows());
  ASSERT_EQ(2, Y.cols());
  MultiVector_double Y_ref(6, 2);
  Y_ref(0,0) = 1; Y_ref(0,1) = 7;
  Y_ref(1,0) = 2; Y_ref(1,1) = 8;
  Y_ref(2,0) = 3; Y_ref(2,1) = 9;
  Y_ref(3,0) = 4; Y_ref(3,1) = 10;
  Y_ref(4,0) = 5; Y_ref(4,1) = 11;
  Y_ref(5,0) = 6; Y_ref(5,1) = 12;
  ASSERT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, single_col_no_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(50,1), Y;

  randomize(X);

  transpose(X, Y);

  ASSERT_EQ(1, Y.rows());
  ASSERT_EQ(50, Y.cols());
  ASSERT_NEAR(ConstEigenMap(X).transpose(), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, single_col_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(50,1), Y;

  randomize(X);

  transpose(X, Y, {25,2});

  ASSERT_EQ(25, Y.rows());
  ASSERT_EQ(2, Y.cols());
  Eigen::MatrixXd Xdata = ConstEigenMap(X);
  const Eigen::MatrixXd Y_ref = Eigen::Map<Eigen::MatrixXd>(Xdata.data(), 2, 25).transpose();
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, single_col_reshape_reverse)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(50,1), Y;

  randomize(X);

  transpose(X, Y, {25,2}, true);

  ASSERT_EQ(25, Y.rows());
  ASSERT_EQ(2, Y.cols());
  Eigen::MatrixXd Xdata = ConstEigenMap(X);
  const Eigen::MatrixXd Y_ref = Eigen::Map<Eigen::MatrixXd>(Xdata.data(), 25, 2);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, large_single_col_no_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(200,1), Y;

  randomize(X);

  transpose(X, Y, {1,200});

  ASSERT_EQ(1, Y.rows());
  ASSERT_EQ(200, Y.cols());
  ASSERT_NEAR(ConstEigenMap(X).transpose(), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, large_single_col_no_reshape_reverse)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(200,1), Y;

  randomize(X);

  transpose(X, Y, {1,200}, true);

  ASSERT_EQ(1, Y.rows());
  ASSERT_EQ(200, Y.cols());
  ASSERT_NEAR(ConstEigenMap(X).transpose(), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, large_single_col_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(200,1), Y;

  randomize(X);

  transpose(X, Y, {100,2});

  ASSERT_EQ(100, Y.rows());
  ASSERT_EQ(2, Y.cols());
  Eigen::MatrixXd Xdata = ConstEigenMap(X);
  const Eigen::MatrixXd Y_ref = Eigen::Map<Eigen::MatrixXd>(Xdata.data(), 2, 100).transpose();
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, large_single_col_reshape_reverse)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(200,1), Y;

  randomize(X);

  transpose(X, Y, {100,2}, true);

  ASSERT_EQ(100, Y.rows());
  ASSERT_EQ(2, Y.cols());
  Eigen::MatrixXd Xdata = ConstEigenMap(X);
  const Eigen::MatrixXd Y_ref = Eigen::Map<Eigen::MatrixXd>(Xdata.data(), 100, 2);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, multi_col_no_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(25,10), Y;

  randomize(X);

  transpose(X, Y);

  ASSERT_EQ(10, Y.rows());
  ASSERT_EQ(25, Y.cols());
  Eigen::MatrixXd Xdata = ConstEigenMap(X);
  const Eigen::MatrixXd Y_ref = Eigen::Map<Eigen::MatrixXd>(Xdata.data(), 25, 10).transpose();
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
  ASSERT_NEAR(ConstEigenMap(X).transpose(), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, multi_col_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(25,10), Y;

  randomize(X);

  transpose(X, Y, {125, 2});

  ASSERT_EQ(125, Y.rows());
  ASSERT_EQ(2, Y.cols());
  Eigen::MatrixXd Xdata = ConstEigenMap(X);
  const Eigen::MatrixXd Y_ref = Eigen::Map<Eigen::MatrixXd>(Xdata.data(), 2, 125).transpose();
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, multi_col_reshape_reverse)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double Y_ref(25,10), X, Y;

  randomize(Y_ref);

  transpose(Y_ref, X, {125, 2});
  transpose(X, Y, {25,10}, true);

  ASSERT_EQ(25, Y.rows());
  ASSERT_EQ(10, Y.cols());
  ASSERT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, LARGE_multi_col_no_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(10000,50), Y;

  randomize(X);

  transpose(X, Y);

  ASSERT_EQ(50, Y.rows());
  ASSERT_EQ(10000, Y.cols());
  Eigen::MatrixXd Xdata = ConstEigenMap(X);
  const Eigen::MatrixXd Y_ref = Eigen::Map<Eigen::MatrixXd>(Xdata.data(), 10000, 50).transpose();
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
  ASSERT_NEAR(ConstEigenMap(X).transpose(), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, LARGE_multi_col_transpose_twice)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double Y_ref(500,7), X, Y;

  randomize(Y_ref);

  transpose(Y_ref, X);
  transpose(X, Y);

  ASSERT_EQ(Y_ref.rows(), Y.rows());
  ASSERT_EQ(Y_ref.cols(), Y.cols());
  ASSERT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, LARGE_multi_col_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(9999,50), Y;

  randomize(X);

  transpose(X, Y, {3333, 150});

  ASSERT_EQ(3333, Y.rows());
  ASSERT_EQ(150, Y.cols());
  Eigen::MatrixXd Xdata = ConstEigenMap(X);
  const Eigen::MatrixXd Y_ref = Eigen::Map<Eigen::MatrixXd>(Xdata.data(), 150, 3333).transpose();
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transpose, consistency_checks)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double Y_ref(112, 21), X, X2, Y;

  randomize(Y_ref);

  // transpose twice == identity
  transpose(Y_ref, X);
  transpose(X, Y);
  EXPECT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);

  // transpose with explicit dimensions (no reshaping)
  transpose(Y_ref, X2, {21, 112}, false);
  EXPECT_NEAR(ConstEigenMap(X), ConstEigenMap(X2), eps);

  // I: transpose and reshape and then reverse reshape and transpose
  transpose(Y_ref, X, {336, 7});
  transpose(X, Y, {112, 21}, true);
  EXPECT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);

  // II: transpose and reshape and then reverse reshape and transpose
  transpose(Y_ref, X, {336, 7}, true);
  transpose(X, Y, {112, 21}, false);
  EXPECT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);

  // III: transpose and reshape and then reverse reshape and transpose
  transpose(Y_ref, X, {56, 42}, false);
  transpose(X, Y, {112, 21}, true);
  EXPECT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);

  // IV: transpose and reshape and then reverse reshape and transpose
  transpose(Y_ref, X, {56, 42}, true);
  transpose(X, Y, {112, 21}, false);
  EXPECT_NEAR(ConstEigenMap(Y_ref), ConstEigenMap(Y), eps);
}
