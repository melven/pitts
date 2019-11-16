#include <gtest/gtest.h>
#include "pitts_tensortrain_laplace_operator.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_random.hpp"

// anonymous namespace
namespace
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;
}


TEST(PITTS_TensorTrain_laplace_operator, rank_1_vector)
{
  TensorTrain_double TT(1,5);

  TT.setZero();
  double norm = laplaceOperator(TT);
  EXPECT_EQ(0, norm2(TT));

  TT.setOnes();
  norm = laplaceOperator(TT);
  EXPECT_NEAR(-1./6, norm*TT.subTensors()[0](0,0,0), eps);
  EXPECT_NEAR(0., norm*TT.subTensors()[0](0,1,0), eps);
  EXPECT_NEAR(0., norm*TT.subTensors()[0](0,2,0), eps);
  EXPECT_NEAR(0., norm*TT.subTensors()[0](0,3,0), eps);
  EXPECT_NEAR(-1./6, norm*TT.subTensors()[0](0,4,0), eps);

  randomize(TT);
  std::array<double, 7> oldVec;
  oldVec[0] = 0.;
  for(int i = 0; i < 5; i++)
    oldVec[i+1] = TT.subTensors()[0](0,i,0);
  oldVec[6] = 0.;
  norm = laplaceOperator(TT);
  for(int i = 0; i < 5; i++)
  {
    EXPECT_NEAR(1./6*oldVec[i]-2./6*oldVec[i+1]+1./6*oldVec[i+2], norm*TT.subTensors()[0](0,i,0), eps);
  }
}

TEST(PITTS_TensorTrain_laplace_operator, rank_2_matrix)
{
  TensorTrain_double TT({3,5},1);

  TT.setZero();
  double norm = laplaceOperator(TT);
  EXPECT_EQ(0, norm2(TT));

  TT.setOnes();
  norm = laplaceOperator(TT);
  TensorTrain_double testTT({3,5},1);
  for(int i = 0; i < 3; i++)
  {
    for(int j = 0; j < 5; j++)
    {
      double refResult = 0;
      if( i == 0 || i+1 == 3 )
        refResult += -1./4;
      if( j == 0 || j+1 == 5 )
        refResult += -1./6;
      testTT.setUnit({i,j});
      EXPECT_NEAR(refResult, norm*dot(TT,testTT), eps);
    }
  }

  TT.setTTranks({2});
  randomize(TT);
  std::array<std::array<double, 7>,5> oldMat;
  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 7; j++)
    {
      if( i > 0 && i+1 < 5 && j > 0 && j+1 < 7 )
      {
        testTT.setUnit({i-1,j-1});
        oldMat[i][j] = dot(TT,testTT);
      }
      else
        oldMat[i][j] = 0.;
    }
  }
  norm = laplaceOperator(TT);
  for(int i = 0; i < 3; i++)
  {
    for(int j = 0; j < 5; j++)
    {
      double refResult = 1./4*oldMat[i][j+1] - 2./4*oldMat[i+1][j+1] + 1./4*oldMat[i+2][j+1]
                       + 1./6*oldMat[i+1][j] - 2./6*oldMat[i+1][j+1] + 1./6*oldMat[i+1][j+2];
      testTT.setUnit({i,j});
      EXPECT_NEAR(refResult, norm*dot(TT,testTT), eps);
    }
  }
}

TEST(PITTS_TensorTrain_laplace_operator, rank_3_cube)
{
  TensorTrain_double TT({2,3,10},1);

  TT.setZero();
  double norm = laplaceOperator(TT);
  EXPECT_EQ(0, norm2(TT));

  TT.setOnes();
  norm = laplaceOperator(TT);
  TensorTrain_double testTT({2,3,10},1);
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 10; k++)
      {
        double refResult = 0;
        if (i == 0 || i + 1 == 2)
          refResult += -1. / 3;
        if (j == 0 || j + 1 == 3)
          refResult += -1. / 4;
        if (k == 0 || k + 1 == 10)
          refResult += -1./11;
        testTT.setUnit({i, j, k});
        EXPECT_NEAR(refResult, norm * dot(TT, testTT), eps);
      }

  TT.setTTranks({2,2});
  randomize(TT);
  std::array<std::array<std::array<double, 12>, 5>,4> oldMat;
  for(int i = 0; i < 4; i++)
    for(int j = 0; j < 5; j++)
      for (int k = 0; k < 12; k++)
      {
        if (i > 0 && i + 1 < 4 && j > 0 && j + 1 < 5 && k > 0 && k + 1 < 12)
        {
          testTT.setUnit({i - 1, j - 1, k - 1});
          oldMat[i][j][k] = dot(TT, testTT);
        }
        else
          oldMat[i][j][k] = 0.;
      }

  norm = laplaceOperator(TT);
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 10; k++)
      {
        double refResult = 1./3*oldMat[i][j+1][k+1] - 2./3*oldMat[i+1][j+1][k+1] + 1./3*oldMat[i+2][j+1][k+1]
                         + 1./4*oldMat[i+1][j][k+1] - 2./4*oldMat[i+1][j+1][k+1] + 1./4*oldMat[i+1][j+2][k+1]
                         + 1./11*oldMat[i+1][j+1][k] - 2./11*oldMat[i+1][j+1][k+1] + 1./11*oldMat[i+1][j+1][k+2];
        testTT.setUnit({i,j,k});
        EXPECT_NEAR(refResult, norm*dot(TT,testTT), eps);
      }
}