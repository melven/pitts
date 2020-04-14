#include <gtest/gtest.h>
#include "pitts_tensortrain.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_random.hpp"

namespace
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  using mat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
  constexpr auto eps = 1.e-10;

  // helper function to contract a subtensor of the tensor train
  Tensor2_double leftContract(const Tensor3_double& subT)
  {
    const auto r1 = subT.r1();
    const auto r2 = subT.r2();
    const auto n = subT.n();

    Tensor2_double result(r2,r2);
    for(int j = 0; j < r2; j++)
      for(int i = 0; i < r2; i++)
        result(i,j) = 0;
    for(int k = 0; k < n; k++)
      for(int j = 0; j < r2; j++)
        for(int i = 0; i < r2; i++)
          for(int i_ = 0; i_ < r1; i_++)
            result(i,j) += subT(i_,k,i)*subT(i_,k,j);
    return result;
  }

  // helper function to call normalize on a tensor train and do common checks
  void check_normalize(TensorTrain_double& TT)
  {
    const TensorTrain_double refTT = TT;
    const double TTnorm = normalize(TT);

    EXPECT_NEAR(norm2(refTT), TTnorm, eps);
    EXPECT_NEAR(1., norm2(TT), eps);

    // check orthogonality of subtensors
    for(const auto& subT: TT.subTensors())
    {
      const mat orthogErr = ConstEigenMap(leftContract(subT)) - mat::Identity(subT.r2(),subT.r2());
      EXPECT_NEAR(0., orthogErr.norm(), eps);
    }

    // check tensor is the same, except for scaling...
    TensorTrain_double testTT(TT.dimensions());
    if( TT.dimensions().size() == 1 )
    {
      for(int i = 0; i < TT.dimensions()[0]; i++)
      {
        testTT.setUnit({i});
        EXPECT_NEAR(dot(refTT,testTT), TTnorm*dot(TT,testTT), eps);
      }
    }
    else if( TT.dimensions().size() == 2 )
    {
      for(int i = 0; i < TT.dimensions()[0]; i++)
        for(int j = 0; j < TT.dimensions()[1]; j++)
        {
          testTT.setUnit({i,j});
          EXPECT_NEAR(dot(refTT,testTT), TTnorm*dot(TT,testTT), eps);
        }
    }
    else
    {
      for(int i = 0; i < 10; i++)
      {
        randomize(testTT);
        EXPECT_NEAR(dot(refTT,testTT), TTnorm*dot(TT,testTT), eps*norm2(testTT));
      }
    }
  }
}

TEST(PITTS_TensorTrain_normalize, unit_vector)
{
  TensorTrain_double TT(1,5);
  TT.setUnit({1});
  check_normalize(TT);
}

TEST(PITTS_TensorTrain_normalize, one_vector)
{
  TensorTrain_double TT(1,5);
  TT.setOnes();
  check_normalize(TT);
}

TEST(PITTS_TensorTrain_normalize, random_vector)
{
  TensorTrain_double TT(1,5);
  randomize(TT);
  check_normalize(TT);
}

TEST(PITTS_TensorTrain_normalize, unit_matrix)
{
  TensorTrain_double TT(2,5);
  TT.setUnit({1,2});
  check_normalize(TT);
}

TEST(PITTS_TensorTrain_normalize, rank_1_ones_matrix)
{
  TensorTrain_double TT(2,5);
  TT.setOnes();
  check_normalize(TT);
}

TEST(PITTS_TensorTrain_normalize, rank_1_random_matrix)
{
  TensorTrain_double TT(2,5);
  randomize(TT);
  check_normalize(TT);
}

TEST(PITTS_TensorTrain_normalize, rank_3_ones_matrix)
{
  TensorTrain_double TT(2,5,3);
  for(auto& subT: TT.editableSubTensors())
    subT.setConstant(1.);
  check_normalize(TT);
  EXPECT_EQ(std::vector<int>({1}), TT.getTTranks());
}

TEST(PITTS_TensorTrain_normalize, rank_3_random_matrix)
{
  TensorTrain_double TT(2,5,3);
  randomize(TT);
  check_normalize(TT);
}

TEST(PITTS_TensorTrain_normalize, rank_1_larger_ones_tensor)
{
  TensorTrain_double TT({4,3,2,5});
  TT.setOnes();
  check_normalize(TT);
}

TEST(PITTS_TensorTrain_normalize, rank_1_larger_random_tensor)
{
  TensorTrain_double TT({4,3,2,5});
  randomize(TT);
  check_normalize(TT);
}

TEST(PITTS_TensorTrain_normalize, larger_ones_tensor)
{
  TensorTrain_double TT({4,3,2,5});
  TT.setTTranks({2,3,4});
  for(auto& subT: TT.editableSubTensors())
    subT.setConstant(1.);
  check_normalize(TT);
  EXPECT_EQ(std::vector<int>({1,1,1}), TT.getTTranks());
}

TEST(PITTS_TensorTrain_normalize, larger_random_tensor)
{
  TensorTrain_double TT({4,3,2,5});
  TT.setTTranks({2,3,4});
  randomize(TT);
  check_normalize(TT);
}

TEST(PITTS_TensorTrain_normalize, even_larger_ones_tensor)
{
  TensorTrain_double TT({50,30,10}, 1);
  TT.setTTranks({5,10});
  for(auto& subT: TT.editableSubTensors())
    subT.setConstant(1.);
  check_normalize(TT);
  EXPECT_EQ(std::vector<int>({1,1}), TT.getTTranks());
}

TEST(PITTS_TensorTrain_normalize, even_larger_random_tensor)
{
  TensorTrain_double TT({50,30,10}, 1);
  TT.setTTranks({5,10});
  randomize(TT);
  check_normalize(TT);
}
