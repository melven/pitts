#include <gtest/gtest.h>
#include "pitts_tensor2_qb_decomposition.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"

TEST(PITTS_Tensor2_qb_decomposition, diagonal_full_rank)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixXd;

  Tensor2_double t2(5,5);

  // make diagonal non-singular matrix
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 5; j++)
      t2(i,j) = (i==j) ? i+4 : 0;

  Tensor2_double B;
  Tensor2_double Binv;
  constexpr double eps = 1.e-10;
  int rank = qb_decomposition(t2, B, Binv, eps);
  ASSERT_EQ(5, rank);

  const auto mapT2 = ConstEigenMap(t2);
  const auto mapB = ConstEigenMap(B);
  const auto mapBinv = ConstEigenMap(Binv);

  const mat invErr = mapB*mapBinv - mat::Identity(5,5);
  ASSERT_NEAR(0, invErr.norm(), eps);

  const mat qbErr = mapBinv.transpose() * mapT2 - mapB;
  ASSERT_NEAR(0, qbErr.norm(), eps);
}

TEST(PITTS_Tensor2_qb_decomposition, diagonal_rank_deficient)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixXd;

  Tensor2_double t2(5,5);

  // make random non-singular spd matrix
  // make diagonal non-singular matrix
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 5; j++)
      t2(i,j) = (i==j) ? i+4 : 0;
  t2(2,2) = 0;

  Tensor2_double B;
  Tensor2_double Binv;
  constexpr double eps = 1.e-10;
  int rank = qb_decomposition(t2, B, Binv, eps);
  ASSERT_EQ(4, rank);

  const auto mapT2 = ConstEigenMap(t2);
  const auto mapB = ConstEigenMap(B);
  const auto mapBinv = ConstEigenMap(Binv);

  mat invErr = mapB*mapBinv;
  invErr.topLeftCorner(4,4) -= mat::Identity(4,4);
  ASSERT_NEAR(0, invErr.norm(), eps);

  const mat qbErr = mapBinv.transpose() * mapT2 - mapB;
  ASSERT_NEAR(0, qbErr.norm(), eps);
}


TEST(PITTS_Tensor2_qb_decomposition, random_full_rank)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixXd;

  Tensor2_double t2(20,20);

  // make random non-singular spd matrix
  {
    mat tmp = mat::Random(20,20);
    EigenMap(t2) = tmp.transpose() * tmp;
    for(int i = 0; i < t2.r1(); i++)
      for(int j = 0; j < t2.r2(); j++)
        t2(i,j) += 20;
  }

  Tensor2_double B;
  Tensor2_double Binv;
  constexpr double eps = 1.e-10;
  int rank = qb_decomposition(t2, B, Binv, eps);
  ASSERT_EQ(20, rank);

  const auto mapT2 = ConstEigenMap(t2);
  const auto mapB = ConstEigenMap(B);
  const auto mapBinv = ConstEigenMap(Binv);

  const mat invErr = mapB*mapBinv - mat::Identity(20,20);
  ASSERT_NEAR(0, invErr.norm(), eps);

  const mat qbErr = mapBinv.transpose() * mapT2 - mapB;
  ASSERT_NEAR(0, qbErr.norm(), eps);
}

TEST(PITTS_Tensor2_qb_decomposition, random_rank_deficient)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixXd;

  Tensor2_double t2(5,5);

  // make random non-singular spd matrix
  {
    mat tmp = mat::Random(5,5);
    EigenMap(t2) = tmp.transpose() * tmp;
    for(int i = 0; i < t2.r1(); i++)
      for(int j = 0; j < t2.r2(); j++)
      {
        if( i == 3 || j == 3 )
          t2(i,j) = 0;
        else
          t2(i,j) += 20;
      }
  }

  Tensor2_double B;
  Tensor2_double Binv;
  constexpr double eps = 1.e-10;
  int rank = qb_decomposition(t2, B, Binv, eps);
  ASSERT_EQ(4, rank);

  const auto mapT2 = ConstEigenMap(t2);
  const auto mapB = ConstEigenMap(B);
  const auto mapBinv = ConstEigenMap(Binv);

  mat invErr = mapB*mapBinv;
  invErr.topLeftCorner(4,4) -= mat::Identity(4,4);
  ASSERT_NEAR(0, invErr.norm(), eps);

  const mat qbErr = mapBinv.transpose() * mapT2 - mapB;
  ASSERT_NEAR(0, qbErr.norm(), eps);
}

TEST(PITTS_Tensor2_qb_decomposition, zero)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixXd;

  Tensor2_double t2(5,5);

  // make random non-singular spd matrix
  EigenMap(t2) = mat::Zero(5,5);


  Tensor2_double B;
  Tensor2_double Binv;
  constexpr double eps = 1.e-10;
  int rank = qb_decomposition(t2, B, Binv, eps, 999, true);
  ASSERT_EQ(0, rank);

  const auto mapB = ConstEigenMap(B);
  const auto mapBinv = ConstEigenMap(Binv);

  ASSERT_NEAR(0, mapB.norm(), eps);
  ASSERT_NEAR(0, mapBinv.norm(), eps);
}

TEST(PITTS_Tensor2_qb_decomposition, absoluteTolerance)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixXd;

  // example matrix to orthognalize:
  mat X = mat::Zero(10,4);
  X(0,0) = 1000;
  X(1,1) = 1.1e-4;
  X(2,2) = 0.9e-4;
  X(3,3) = 1.1e-8;

  Tensor2_double XtX(4,4);
  EigenMap(XtX) = X.transpose()*X;
  Tensor2_double B, Binv;

  int rank = qb_decomposition(XtX, B, Binv, 1., 999, true);
  EXPECT_EQ(1, rank);

  mat invErr = ConstEigenMap(B) * ConstEigenMap(Binv);
  invErr.topLeftCorner(1,1) -= mat::Identity(1,1);
  EXPECT_NEAR(mat::Zero(1,1), invErr, 1.e-13);
  mat qbErr = ConstEigenMap(Binv).transpose() * ConstEigenMap(XtX) - ConstEigenMap(B);
  EXPECT_NEAR(mat::Zero(1,4), qbErr, 1.e-13);

  rank = qb_decomposition(XtX, B, Binv, 1.e-4, 999, true);
  EXPECT_EQ(2, rank);

  invErr = ConstEigenMap(B) * ConstEigenMap(Binv);
  invErr.topLeftCorner(2,2) -= mat::Identity(2,2);
  EXPECT_NEAR(mat::Zero(2,2), invErr, 1.e-13);
  qbErr = ConstEigenMap(Binv).transpose() * ConstEigenMap(XtX) - ConstEigenMap(B);
  EXPECT_NEAR(mat::Zero(2,4), qbErr, 1.e-13);

  rank = qb_decomposition(XtX, B, Binv, 1.e-8, 999, true);
  EXPECT_EQ(4, rank);

  invErr = ConstEigenMap(B) * ConstEigenMap(Binv);
  invErr.topLeftCorner(4,4) -= mat::Identity(4,4);
  EXPECT_NEAR(mat::Zero(4,4), invErr, 1.e-13);
  qbErr = ConstEigenMap(Binv).transpose() * ConstEigenMap(XtX) - ConstEigenMap(B);
  EXPECT_NEAR(mat::Zero(4,4), qbErr, 1.e-13);
}

TEST(PITTS_Tensor2_qb_decomposition, relativeTolerance)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixXd;

  // example matrix to orthognalize:
  mat X = mat::Zero(10,4);
  X(0,0) = 1000;
  X(1,1) = 1.1e-4;
  X(2,2) = 0.9e-4;
  X(3,3) = 1.1e-8;

  Tensor2_double XtX(4,4);
  EigenMap(XtX) = X.transpose()*X;
  Tensor2_double B, Binv;

  int rank = qb_decomposition(XtX, B, Binv, 1.e-3, 999, false);
  EXPECT_EQ(1, rank);

  mat invErr = ConstEigenMap(B) * ConstEigenMap(Binv);
  invErr.topLeftCorner(1,1) -= mat::Identity(1,1);
  EXPECT_NEAR(mat::Zero(1,1), invErr, 1.e-13);
  mat qbErr = ConstEigenMap(Binv).transpose() * ConstEigenMap(XtX) - ConstEigenMap(B);
  EXPECT_NEAR(mat::Zero(1,4), qbErr, 1.e-13);

  rank = qb_decomposition(XtX, B, Binv, 1.e-7, 999, false);
  EXPECT_EQ(2, rank);

  invErr = ConstEigenMap(B) * ConstEigenMap(Binv);
  invErr.topLeftCorner(2,2) -= mat::Identity(2,2);
  EXPECT_NEAR(mat::Zero(2,2), invErr, 1.e-13);
  qbErr = ConstEigenMap(Binv).transpose() * ConstEigenMap(XtX) - ConstEigenMap(B);
  EXPECT_NEAR(mat::Zero(2,4), qbErr, 1.e-13);

  rank = qb_decomposition(XtX, B, Binv, 1.e-11, 999, false);
  EXPECT_EQ(4, rank);

  invErr = ConstEigenMap(B) * ConstEigenMap(Binv);
  invErr.topLeftCorner(4,4) -= mat::Identity(4,4);
  EXPECT_NEAR(mat::Zero(4,4), invErr, 1.e-13);
  qbErr = ConstEigenMap(Binv).transpose() * ConstEigenMap(XtX) - ConstEigenMap(B);
  EXPECT_NEAR(mat::Zero(4,4), qbErr, 1.e-13);
}
