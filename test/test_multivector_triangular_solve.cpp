#include <gtest/gtest.h>
#include "pitts_multivector_triangular_solve.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"


// anonymous namespace
namespace
{
  const auto eps = 1.e-8;

  template<typename T>
  void test_triangularSolve(PITTS::MultiVector<T>& X, const PITTS::Tensor2<T>& R, const std::vector<int>& colsPermutation)
  {
    const Eigen::MatrixX<T> X_input = ConstEigenMap(X);
    Eigen::MatrixX<T> X_ref;
    const auto mapR = ConstEigenMap(R).template triangularView<Eigen::Upper>();

    if( colsPermutation.empty() )
      X_ref = X_input;
    else
      X_ref = X_input(Eigen::placeholders::all, colsPermutation);

    X_ref = mapR.template solve<Eigen::OnTheRight>(X_ref);

    triangularSolve(X, R, colsPermutation);

    EXPECT_NEAR(X_ref, ConstEigenMap(X), eps);

    if( colsPermutation.empty() )
    {
      EXPECT_NEAR(X_input, ConstEigenMap(X) * mapR, eps);
    }
  }
}

TEST(PITTS_MultiVector_triangularSolve, invalid_args)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(50,3);
  Tensor2_double R(1,1);

  EXPECT_THROW(triangularSolve(X, R), std::invalid_argument);

  R.resize(3,2);

  EXPECT_THROW(triangularSolve(X, R), std::invalid_argument);

  R.resize(3,3);
  randomize(X);
  randomize(R);

  EXPECT_NO_THROW(triangularSolve(X, R));

  EXPECT_THROW(triangularSolve(X, R, {0, 1, 2, 3}), std::invalid_argument);

  EXPECT_NO_THROW(triangularSolve(X, R, {1, 0, 2}));

  EXPECT_THROW(triangularSolve(X, R, {5, 1, 2}), std::invalid_argument);
}

TEST(PITTS_MultiVector_triangularSolve, single_col)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(20,1);
  randomize(X);
  Tensor2_double R(1,1);
  R(0,0) = 1.7;

  test_triangularSolve(X, R, {});
}

TEST(PITTS_MultiVector_triangularSolve, large_single_col)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(100,1);
  randomize(X);
  Tensor2_double R(1,1);
  R(0,0) = 1.7;

  test_triangularSolve(X, R, {});
}

TEST(PITTS_MultiVector_triangularSolve, small_two_cols_diagonal)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(5,2);
  randomize(X);
  Tensor2_double R(2,2);
  R(0,0) = 2;
  R(0,1) = 0;
  R(1,1) = -1;

  test_triangularSolve(X, R, {});
}

TEST(PITTS_MultiVector_triangularSolve, small_two_cols)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(3,2);
  X(0,0) = 1; X(0,1) = 10;
  X(1,0) = 2; X(1,1) = 11;
  X(2,0) = 3; X(2,1) = 12;
  Tensor2_double R(2,2);
  R(0,0) = 0.5;
  R(0,1) = 1;
  R(1,1) = -1;

  test_triangularSolve(X, R, {});
}

TEST(PITTS_MultiVector_triangularSolve, small_two_cols_pivoted)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(3,2);
  X(0,0) = 1; X(0,1) = 10;
  X(1,0) = 2; X(1,1) = 11;
  X(2,0) = 3; X(2,1) = 12;
  Tensor2_double R(2,2);
  R(0,0) = 0.5;
  R(0,1) = 1;
  R(1,1) = -1;

  test_triangularSolve(X, R, {1,0});
}

TEST(PITTS_MultiVector_triangularSolve, small_two_cols_pivoted_rankDeficient)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(3,2);
  X(0,0) = 1; X(0,1) = 10;
  X(1,0) = 2; X(1,1) = 11;
  X(2,0) = 3; X(2,1) = 12;
  Tensor2_double R(1,1);
  R(0,0) = 0.5;

  test_triangularSolve(X, R, {1});
}

TEST(PITTS_MultiVector_triangularSolve, two_cols)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(20,2);
  randomize(X);
  Tensor2_double R(2,2);
  R(0,0) = 1.7;
  R(0,1) = 0.3;
  R(1,1) = 0.8;

  test_triangularSolve(X, R, {});
}

TEST(PITTS_MultiVector_triangularSolve, large_two_cols)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(75,2);
  randomize(X);
  Tensor2_double R(2,2);
  R(0,0) = 1.3;
  R(0,1) = -0.2;
  R(1,1) = 0.93;

  test_triangularSolve(X, R, {});
}

TEST(PITTS_MultiVector_triangularSolve, two_cols_pivoted)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(20,2);
  randomize(X);
  Tensor2_double R(2,2);
  R(0,0) = 1.7;
  R(0,1) = 0.3;
  R(1,1) = 0.8;

  test_triangularSolve(X, R, {1,0});
}

TEST(PITTS_MultiVector_triangularSolve, random_nx20)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(50,20);
  randomize(X);
  Tensor2_double R(20,20);
  randomize(R);
  for(int i = 0; i < 20; i++)
    R(i,i) += 3;

  test_triangularSolve(X, R, {});
}

TEST(PITTS_MultiVector_triangularSolve, random_nx5_pivoted)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(30,5);
  randomize(X);
  Tensor2_double R(5,5);
  randomize(R);
  for(int i = 0; i < 5; i++)
    R(i,i) += 3;

  test_triangularSolve(X, R, {3,1,0,2,4});
}

TEST(PITTS_MultiVector_triangularSolve, random_nx5_pivoted_rankDeficient)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(30,5);
  randomize(X);
  Tensor2_double R(3,3);
  randomize(R);
  for(int i = 0; i < 3; i++)
    R(i,i) += 3;

  test_triangularSolve(X, R, {3,1,0});
}

TEST(PITTS_MultiVector_triangularSolve, large_random_nx5_pivoted)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(200,5);
  randomize(X);
  Tensor2_double R(5,5);
  randomize(R);
  for(int i = 0; i < 5; i++)
    R(i,i) += 3;

  test_triangularSolve(X, R, {3,1,0,2,4});
}

TEST(PITTS_MultiVector_triangularSolve, large_random_nx5_pivoted_rankDeficient)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(200,5);
  randomize(X);
  Tensor2_double R(3,3);
  randomize(R);
  for(int i = 0; i < 3; i++)
    R(i,i) += 3;

  test_triangularSolve(X, R, {3,1,0});
}

TEST(PITTS_MultiVector_triangularSolve, random_QR)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixXd;

  MultiVector_double X(20,8);
  randomize(X);
  for(int i = 0; i < X.cols(); i++)
    X(i,i) += 3;
  EigenMap(X).col(5).setZero();
  Eigen::ColPivHouseholderQR<mat> qr(ConstEigenMap(X));
  EXPECT_EQ(7, qr.rank());
  Tensor2_double R(qr.rank(), qr.rank());
  EigenMap(R) = qr.matrixR().topLeftCorner(qr.rank(), qr.rank()).template triangularView<Eigen::Upper>();
  std::vector<int> colPerm(qr.rank());
  Eigen::Map<Eigen::VectorXi>(colPerm.data(), colPerm.size()) = qr.colsPermutation().indices().head(qr.rank());

  test_triangularSolve(X, R, colPerm);
  qr.householderQ().setLength(qr.rank());
  mat Q = qr.householderQ();
  EXPECT_NEAR(Q.leftCols(qr.rank()), ConstEigenMap(X), eps);
}