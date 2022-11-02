#include <gtest/gtest.h>
#include "pitts_tensortrain.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain_axpby_plain.hpp"
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
    double gamma = internal::axpby_plain(alpha, TTa, beta, TTresult);
    const double TTresult_norm = norm2(TTresult);

    // check Law of cosines
    EXPECT_NEAR(pow2(gamma*TTresult_norm), pow2(alpha*TTa_norm)+pow2(beta*TTb_norm) + 2*alpha*beta*dot(TTa,TTb), eps);

    // check that the result is normalized
    // except for the case alpha or beta == 0
    const auto nDim = TTresult.dimensions().size();
    if( alpha*beta != 0 )
    {
      EXPECT_NEAR(1., norm2(TTresult), eps);
      // check orthogonality of subtensors
      for(int iDim = 0; iDim < nDim; iDim++)
      {
        const auto& subT = TTresult.subTensor(iDim);

        if( iDim != nDim-1 )
        {
          const mat orthogErr = ConstEigenMap(leftContract(subT)) - mat::Identity(subT.r2(),subT.r2());
          EXPECT_NEAR(0., orthogErr.norm(), eps);
        }
        else // iDim == nDim-1
        {
          const double normalizeErr = PITTS::internal::t3_nrm(subT) - 1;
          EXPECT_NEAR(0., normalizeErr, eps);
        }
      }
    }

    // check that the result is correct
    TensorTrain_double testTT(TTresult.dimensions());
    copy(TTb, testTT);
    if( nDim == 1 && testTT.subTensor(0).r1() == 1 && testTT.subTensor(nDim-1).r2() == 1 )
    {
      for(int i = 0; i < TTresult.dimensions()[0]; i++)
      {
        testTT.setUnit({i});
        EXPECT_NEAR(alpha*dot(TTa,testTT)+beta*dot(TTb,testTT), gamma*dot(TTresult,testTT), eps);
      }
    }
    else if( nDim == 2 && testTT.subTensor(0).r1() == 1 && testTT.subTensor(nDim-1).r2() == 1 )
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

TEST(PITTS_TensorTrain_axpby_plain, corner_cases_rank1)
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

TEST(PITTS_TensorTrain_axpby_plain, unit_vectors_same_direction)
{
  TensorTrain_double TTa(1,5), TTb(1,5);
  TTa.setUnit({1});
  TTb.setUnit({1});

  auto [gamma, TTc] = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(7., gamma, eps);
}

TEST(PITTS_TensorTrain_axpby_plain, unit_vectors_different_directions)
{
  TensorTrain_double TTa(1,5), TTb(1,5);
  TTa.setUnit({1});
  TTb.setUnit({3});

  auto [gamma, TTc] = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(5., gamma, eps);
}

TEST(PITTS_TensorTrain_axpby_plain, random_vectors)
{
  TensorTrain_double TTa(1,5), TTb(1,5);
  randomize(TTa);
  randomize(TTb);

  check_axpby(4., TTa, 3., TTb);
}

TEST(PITTS_TensorTrain_axpby_plain, rank2_unit_vectors_same_direction)
{
  TensorTrain_double TTa(2,5), TTb(2,5);
  TTa.setUnit({1,3});
  TTb.setUnit({1,3});

  auto [gamma, TTc] = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(7., gamma, eps);
}

TEST(PITTS_TensorTrain_axpby_plain, rank2_unit_vectors_different_directions)
{
  TensorTrain_double TTa(2,5), TTb(2,5);
  TTa.setUnit({1,1});
  TTb.setUnit({3,3});

  auto [gamma, TTc] = check_axpby(4., TTa, 3., TTb);
  EXPECT_NEAR(5., gamma, eps);
}

TEST(PITTS_TensorTrain_axpby_plain, rank2_random_vectors)
{
  TensorTrain_double TTa(2,5,2), TTb(2,5,2);
  randomize(TTa);
  randomize(TTb);

  check_axpby(4., TTa, 3., TTb);
}

TEST(PITTS_TensorTrain_axpby_plain, larger_random_tensor)
{
  TensorTrain_double TTx({4,3,2,5}), TTy({4,3,2,5});
  TTx.setTTranks({2,3,4});
  TTy.setTTranks({1,4,2});
  randomize(TTx);
  randomize(TTy);

  check_axpby(2., TTx, -1.5, TTy);
}

TEST(PITTS_TensorTrain_axpby_plain, even_larger_ones_tensor)
{
  TensorTrain_double TTx({50,30,10}, 1), TTy({50,30,10}, 1);
  TTx.setTTranks({5,10});
  TTy.setTTranks({3,2});
  Tensor3_double newSubT;
  const int nDim = TTx.dimensions().size();
  for(int iDim = 0; iDim < nDim; iDim++)
  {
    newSubT.resize(TTx.subTensor(iDim).r1(), TTx.subTensor(iDim).n(), TTx.subTensor(iDim).r2());
    newSubT.setConstant(1.);
    newSubT = TTx.setSubTensor(iDim, std::move(newSubT));
  }
  for(int iDim = 0; iDim < nDim; iDim++)
  {
    newSubT.resize(TTy.subTensor(iDim).r1(), TTy.subTensor(iDim).n(), TTy.subTensor(iDim).r2());
    newSubT.setConstant(1.);
    newSubT = TTy.setSubTensor(iDim, std::move(newSubT));
  }
  
  auto [gamma,TTc] = check_axpby(0.005, TTx, -0.00015, TTy);
  EXPECT_EQ(std::vector<int>({1,1}), TTc.getTTranks());
}

TEST(PITTS_TensorTrain_axpby_plain, even_larger_random_tensor)
{
  TensorTrain_double TTx({50,30,10}, 1), TTy({50,30,10}, 1);
  TTx.setTTranks({5,10});
  TTy.setTTranks({1,6});
  randomize(TTx);
  randomize(TTy);

  check_axpby(0.002, TTx, -0.0005, TTy);
}

TEST(PITTS_TensorTrain_axpby_plain, approximation_accuracy_d2)
{
  TensorTrain_double TTx({20,30},1), TTy({20,30},1);
  TTx.setTTranks({10});
  TTy.setTTranks({10});

  // construct some interesting subtensors that are orthogonal
  std::vector<Tensor3_double> subTx(2), subTy(2);
  subTx[0].resize(1, 20, 10);
  subTy[0].resize(1, 20, 10);
  for(int i = 0; i < 20; i++)
    for(int j = 0; j < 10; j++)
    {
      subTx[0](0,i,j) = i==j ? 1 : 0;
      subTy[0](0,i,j) = i==j+10 ? 1 : 0;
    }
  subTx[1].resize(10, 30, 1);
  subTy[1].resize(10, 30, 1);
  for(int i = 0; i < 10; i++)
    for(int j = 0; j < 30; j++)
    {
      subTx[1](i,j,0) = 2*i==j ? std::pow(0.5,i) : 0;
      subTy[1](i,j,0) = 2*i+1==j ? std::pow(0.5,i) : 0;
    }
  TTx.setSubTensors(0, std::move(subTx));
  TTy.setSubTensors(0, std::move(subTy));

  TensorTrain_double TTz({20,30},1);

  // accurate addition
  PITTS::copy(TTy, TTz);
  double nrm = PITTS::internal::axpby_plain(1., TTx, 1., TTz);
  EXPECT_EQ(std::vector<int>({20}), TTz.getTTranks());
  nrm = PITTS::internal::axpby_plain(-1., TTy, nrm, TTz);
  nrm = PITTS::internal::axpby_plain(-1., TTx, nrm, TTz);
  EXPECT_NEAR(0, nrm, eps);

  // less accurate addition
  PITTS::copy(TTy, TTz);
  nrm = PITTS::internal::axpby_plain(1., TTx, 1., TTz, 0.1);
  EXPECT_GT(10, TTz.getTTranks()[0]);
  nrm = PITTS::internal::axpby_plain(-1., TTy, nrm, TTz);
  nrm = PITTS::internal::axpby_plain(-1., TTx, nrm, TTz);
  EXPECT_NEAR(0.1, nrm, 0.05);

  // somewhat more accurate addition
  PITTS::copy(TTy, TTz);
  nrm = PITTS::internal::axpby_plain(1., TTx, 1., TTz, 0.01);
  EXPECT_LT(10, TTz.getTTranks()[0]);
  EXPECT_GT(20, TTz.getTTranks()[0]);
  nrm = PITTS::internal::axpby_plain(-1., TTy, nrm, TTz);
  nrm = PITTS::internal::axpby_plain(-1., TTx, nrm, TTz);
  EXPECT_NEAR(0.01, nrm, 0.005);

  // force specific max. rank
  PITTS::copy(TTy, TTz);
  nrm = PITTS::internal::axpby_plain(1., TTx, 1., TTz, 0., 15);
  EXPECT_EQ(15, TTz.getTTranks()[0]);
  nrm = PITTS::internal::axpby_plain(-1., TTy, nrm, TTz);
  nrm = PITTS::internal::axpby_plain(-1., TTx, nrm, TTz);
  EXPECT_NEAR(0, nrm, 0.02);

  // force smaller max. rank
  PITTS::copy(TTy, TTz);
  nrm = PITTS::internal::axpby_plain(1., TTx, 1., TTz, 0., 5);
  EXPECT_EQ(5, TTz.getTTranks()[0]);
  nrm = PITTS::internal::axpby_plain(-1., TTy, nrm, TTz);
  nrm = PITTS::internal::axpby_plain(-1., TTx, nrm, TTz);
  EXPECT_NEAR(0.3, nrm, 0.1);
}

TEST(PITTS_TensorTrain_axpby_plain, boundaryRank_nDim1_constant)
{
  TensorTrain_double TTx(1, 5), TTy(1, 5);

  {
    Tensor3_double subTx(3,5,4);
    Tensor3_double subTy(3,5,4);

    subTx.setConstant(1);
    subTy.setConstant(2);

    TTx.setSubTensor(0, std::move(subTx));
    TTy.setSubTensor(0, std::move(subTy));
  }


  const double nrm = internal::axpby_plain(0.7, TTx, 0.3, TTy);
  const double nrm_ref = std::sqrt(1.3*1.3*3*5*4);
  EXPECT_NEAR(nrm_ref, nrm, eps);
  const double v_ref = 1/std::sqrt(3*5*4.);
  const auto& subTy = TTy.subTensor(0);
  for(int i = 0; i < subTy.r1(); i++)
    for(int j = 0; j < subTy.n(); j++)
      for(int k = 0; k < subTy.r2(); k++)
      {
        EXPECT_NEAR(v_ref, subTy(i,j,k), eps);
      }
}

TEST(PITTS_TensorTrain_axpby_plain, boundaryRank_nDim1_random)
{
  TensorTrain_double TTx(1, 5), TTy(1, 5);
  Tensor3_double subTx(3,5,4);
  Tensor3_double subTy(3,5,4);
  randomize(subTx);
  randomize(subTy);

  TTx.setSubTensor(0, std::move(subTx));
  TTy.setSubTensor(0, std::move(subTy));

  check_axpby(0.2, TTx, -0.5, TTy);
}

TEST(PITTS_TensorTrain_axpby_plain, boundaryRank_nDim3_random)
{
  TensorTrain_double TTx({5,3,2}, 2), TTy({5,3,2}, 2);
  TTx.setTTranks({5,3});
  TTy.setTTranks({1,4});
  TTx.setSubTensor(0, Tensor3_double(3,5,5));
  TTx.setSubTensor(2, Tensor3_double(3,2,3));
  TTy.setSubTensor(0, Tensor3_double(3,5,1));
  TTy.setSubTensor(2, Tensor3_double(4,2,3));
  randomize(TTx);
  randomize(TTy);

  check_axpby(0.2, TTx, -0.5, TTy);
}
