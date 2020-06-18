#include <gtest/gtest.h>
#include "pitts_fixed_tensortrain.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_fixed_tensor3.hpp"
#include "pitts_fixed_tensortrain_axpby.hpp"
#include "pitts_fixed_tensortrain_dot.hpp"
#include "pitts_fixed_tensortrain_random.hpp"

namespace
{
  template<typename T>
  auto pow2(T x)
  {
    return x*x;
  }

  static constexpr auto DIM = 5;

  using FixedTensorTrain_double = PITTS::FixedTensorTrain<double, DIM>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using FixedTensor3_double = PITTS::FixedTensor3<double, DIM>;
  using mat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
  constexpr auto eps = 1.e-10;

  double norm2(const FixedTensorTrain_double& TTv)
  {
    return std::sqrt(dot(TTv, TTv));
  }

  // helper function to contract a subtensor of the tensor train
  Tensor2_double leftContract(const FixedTensor3_double& subT)
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
  auto check_axpby(double alpha, const FixedTensorTrain_double& TTa, double beta, const FixedTensorTrain_double& TTb)
  {
    const double TTa_norm = norm2(TTa);
    const double TTb_norm = norm2(TTb);

    FixedTensorTrain_double TTresult = TTb;
    axpby(alpha, TTa, beta, TTresult);
    const double TTresult_norm = norm2(TTresult);

    // check Law of cosines
    EXPECT_NEAR(pow2(TTresult_norm), pow2(alpha*TTa_norm)+pow2(beta*TTb_norm) + 2*alpha*beta*dot(TTa,TTb), eps);

    // check that the result is normalized
    // except for the case alpha or beta == 0
    if( alpha*beta != 0 )
    {
      // check orthogonality of subtensors
      for(int iDim = 0; iDim < TTresult.nDims(); iDim++)
      {
        const auto& subT = TTresult.subTensors()[iDim];
        mat ref;
        if( iDim < TTresult.nDims()-1 )
          ref = mat::Identity(subT.r2(), subT.r2());
        else
          ref = mat::Constant(1,1, pow2(TTresult_norm));
        const mat orthogErr = ConstEigenMap(leftContract(subT)) - ref;
        EXPECT_NEAR(0., orthogErr.norm(), eps);
      }
    }

    // check that the result is correct
    FixedTensorTrain_double testTT(TTresult.nDims());
    if( TTresult.nDims() == 1 )
    {
      for(int i = 0; i < DIM; i++)
      {
        testTT.setUnit({i});
        EXPECT_NEAR(alpha*dot(TTa,testTT)+beta*dot(TTb,testTT), dot(TTresult,testTT), eps);
      }
    }
    else if( TTresult.nDims() == 2 )
    {
      for(int i = 0; i < DIM; i++)
        for(int j = 0; j < DIM; j++)
        {
          testTT.setUnit({i,j});
          EXPECT_NEAR(alpha * dot(TTa, testTT) + beta * dot(TTb, testTT), dot(TTresult, testTT), eps);
        }
    }
    else
    {
      for(int i = 0; i < 10; i++)
      {
        randomize(testTT);
        EXPECT_NEAR(alpha * dot(TTa, testTT) + beta * dot(TTb, testTT), dot(TTresult, testTT), eps);
      }
    }

    return TTresult;
  }
}

TEST(PITTS_FixedTensorTrain_axpby, corner_cases_rank1)
{
  FixedTensorTrain_double TTa(1), TTb(1);
  randomize(TTa);
  randomize(TTb);
  check_axpby(0, TTa, 0, TTb);
  check_axpby(0, TTa, 1, TTb);
  check_axpby(0, TTa, 3, TTb);
  check_axpby(1, TTa, 0, TTb);
  check_axpby(3, TTa, 0, TTb);
}

TEST(PITTS_FixedTensorTrain_axpby, unit_vectors_same_direction)
{
  FixedTensorTrain_double TTa(1), TTb(1);
  TTa.setUnit({1});
  TTb.setUnit({1});

  const auto TTc = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(7., norm2(TTc), eps);
}

TEST(PITTS_FixedTensorTrain_axpby, unit_vectors_different_directions)
{
  FixedTensorTrain_double TTa(1), TTb(1);
  TTa.setUnit({1});
  TTb.setUnit({3});

  const auto TTc = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(5., norm2(TTc), eps);
}

TEST(PITTS_FixedTensorTrain_axpby, random_vectors)
{
  FixedTensorTrain_double TTa(1), TTb(1);
  randomize(TTa);
  randomize(TTb);

  check_axpby(4., TTa, 3., TTb);
}

TEST(PITTS_FixedTensorTrain_axpby, rank2_unit_vectors_same_direction)
{
  FixedTensorTrain_double TTa(2), TTb(2);
  TTa.setUnit({1,3});
  TTb.setUnit({1,3});

  const auto TTc = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(7., norm2(TTc), eps);
}

TEST(PITTS_FixedTensorTrain_axpby, rank2_unit_vectors_different_directions)
{
  FixedTensorTrain_double TTa(2), TTb(2);
  TTa.setUnit({1,1});
  TTb.setUnit({3,3});

  const auto TTc = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(5., norm2(TTc), eps);
}

TEST(PITTS_FixedTensorTrain_axpby, rank2_random_vectors)
{
  FixedTensorTrain_double TTa(2,2), TTb(2,2);
  randomize(TTa);
  randomize(TTb);

  check_axpby(4., TTa, 3., TTb);
}

TEST(PITTS_FixedTensorTrain_axpby, larger_random_tensor)
{
  FixedTensorTrain_double TTx(4), TTy(4);
  TTx.setTTranks({2,3,4});
  TTy.setTTranks({1,4,2});
  randomize(TTx);
  randomize(TTy);

  check_axpby(2., TTx, -1.5, TTy);
}
