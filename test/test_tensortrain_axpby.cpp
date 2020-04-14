#include <gtest/gtest.h>
#include "pitts_tensortrain.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_random.hpp"

namespace
{
  template<typename T>
  auto pow2(T x)
  {
    return x*x;
  }

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

  // helper function to call axpby on a tensor train and do common checks
  auto check_axpby(double alpha, const TensorTrain_double& TTa, double beta, const TensorTrain_double& TTb)
  {
    const double TTa_norm = norm2(TTa);
    const double TTb_norm = norm2(TTb);

    TensorTrain_double TTresult = TTb;
    double gamma = axpby(alpha, TTa, beta, TTresult);
    const double TTresult_norm = norm2(TTresult);

    // check Law of cosines
    EXPECT_NEAR(pow2(gamma*TTresult_norm), pow2(alpha*TTa_norm)+pow2(beta*TTb_norm) + 2*alpha*beta*dot(TTa,TTb), eps);

    // check that the result is normalized
    // except for the case alpha or beta == 0
    if( alpha*beta != 0 )
    {
      EXPECT_NEAR(1., norm2(TTresult), eps);
      // check orthogonality of subtensors
      for(const auto& subT: TTresult.subTensors())
      {
        const mat orthogErr = ConstEigenMap(leftContract(subT)) - mat::Identity(subT.r2(),subT.r2());
        EXPECT_NEAR(0., orthogErr.norm(), eps);
      }
    }

    // check that the result is correct
    TensorTrain_double testTT(TTresult.dimensions());
    if( TTresult.dimensions().size() == 1 )
    {
      for(int i = 0; i < TTresult.dimensions()[0]; i++)
      {
        testTT.setUnit({i});
        EXPECT_NEAR(alpha*dot(TTa,testTT)+beta*dot(TTb,testTT), gamma*dot(TTresult,testTT), eps);
      }
    }
    else if( TTresult.dimensions().size() == 2 )
    {
      for(int i = 0; i < TTresult.dimensions()[0]; i++)
        for(int j = 0; j < TTresult.dimensions()[1]; j++)
        {
          testTT.setUnit({i,j});
          EXPECT_NEAR(alpha * dot(TTa, testTT) + beta * dot(TTb, testTT), gamma * dot(TTresult, testTT), eps);
        }
    }
    else
    {
      for(int i = 0; i < 10; i++)
      {
        randomize(testTT);
        EXPECT_NEAR(alpha * dot(TTa, testTT) + beta * dot(TTb, testTT), gamma * dot(TTresult, testTT), eps);
      }
    }

    return std::make_pair(gamma, TTresult);
  }
}

TEST(PITTS_TensorTrain_axpby, corner_cases_rank1)
{
  TensorTrain_double TTa(1,5), TTb(1,5);
  randomize(TTa);
  randomize(TTb);
  check_axpby(0, TTa, 0, TTb);
  check_axpby(0, TTa, 1, TTb);
  check_axpby(0, TTa, 3, TTb);
  check_axpby(1, TTa, 0, TTb);
  check_axpby(3, TTa, 0, TTb);
}

TEST(PITTS_TensorTrain_axpby, unit_vectors_same_direction)
{
  TensorTrain_double TTa(1,5), TTb(1,5);
  TTa.setUnit({1});
  TTb.setUnit({1});

  auto [gamma, TTc] = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(7., gamma, eps);
}

TEST(PITTS_TensorTrain_axpby, unit_vectors_different_directions)
{
  TensorTrain_double TTa(1,5), TTb(1,5);
  TTa.setUnit({1});
  TTb.setUnit({3});

  auto [gamma, TTc] = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(5., gamma, eps);
}

TEST(PITTS_TensorTrain_axpby, random_vectors)
{
  TensorTrain_double TTa(1,5), TTb(1,5);
  randomize(TTa);
  randomize(TTb);

  check_axpby(4., TTa, 3., TTb);
}

TEST(PITTS_TensorTrain_axpby, rank2_unit_vectors_same_direction)
{
  TensorTrain_double TTa(2,5), TTb(2,5);
  TTa.setUnit({1,3});
  TTb.setUnit({1,3});

  auto [gamma, TTc] = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(7., gamma, eps);
}

TEST(PITTS_TensorTrain_axpby, rank2_unit_vectors_different_directions)
{
  TensorTrain_double TTa(2,5), TTb(2,5);
  TTa.setUnit({1,1});
  TTb.setUnit({3,3});

  auto [gamma, TTc] = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(5., gamma, eps);
}

TEST(PITTS_TensorTrain_axpby, rank2_random_vectors)
{
  TensorTrain_double TTa(2,5,2), TTb(2,5,2);
  randomize(TTa);
  randomize(TTb);

  check_axpby(4., TTa, 3., TTb);
}

TEST(PITTS_TensorTrain_axpby, larger_random_tensor)
{
  TensorTrain_double TTx({4,3,2,5}), TTy({4,3,2,5});
  TTx.setTTranks({2,3,4});
  TTy.setTTranks({1,4,2});
  randomize(TTx);
  randomize(TTy);

  check_axpby(2., TTx, -1.5, TTy);
}

TEST(PITTS_TensorTrain_axpby, even_larger_ones_tensor)
{
  TensorTrain_double TTx({50,30,10}, 1), TTy({50,30,10}, 1);
  TTx.setTTranks({5,10});
  TTy.setTTranks({3,2});
  for(auto& subT: TTx.editableSubTensors())
    subT.setConstant(1.);
  for(auto& subT: TTy.editableSubTensors())
    subT.setConstant(1.);
  
  auto [gamma,TTc] = check_axpby(0.005, TTx, -0.00015, TTy);
  EXPECT_EQ(std::vector<int>({1,1}), TTc.getTTranks());
}

TEST(PITTS_TensorTrain_axpby, even_larger_random_tensor)
{
  TensorTrain_double TTx({50,30,10}, 1), TTy({50,30,10}, 1);
  TTx.setTTranks({5,10});
  TTy.setTTranks({1,6});
  randomize(TTx);
  randomize(TTy);

  check_axpby(0.002, TTx, -0.0005, TTy);
}
