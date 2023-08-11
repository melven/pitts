#include <gtest/gtest.h>
#include "pitts_tensortrain.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensor3_random.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_random.hpp"
#include "eigen_test_helper.hpp"
#include <random>

namespace
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  using mat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
  constexpr auto eps = 1.e-8;

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

  // helper function to contract a subtensor of the tensor train
  Tensor2_double rightContract(const Tensor3_double& subT)
  {
    const auto r1 = subT.r1();
    const auto r2 = subT.r2();
    const auto n = subT.n();

    Tensor2_double result(r1,r1);
    for(int j = 0; j < r1; j++)
      for(int i = 0; i < r1; i++)
        result(i,j) = 0;
    for(int k = 0; k < n; k++)
      for(int j = 0; j < r1; j++)
        for(int i = 0; i < r1; i++)
          for(int i_ = 0; i_ < r2; i_++)
            result(i,j) += subT(i,k,i_)*subT(j,k,i_);
    return result;
  }

  // helper function to call normalize on a tensor train and do common checks
  void check_normalize(TensorTrain_double& TT)
  {
    const TensorTrain_double refTT = TT;
    const double TTnorm = normalize(TT);

    EXPECT_NEAR(norm2(refTT), TTnorm, eps);
    EXPECT_NEAR(1., norm2(TT), eps);
    EXPECT_NE(PITTS::TT_Orthogonality::none, TT.isOrthogonal());

    // check orthogonality of subtensors
    const int nDim = TT.dimensions().size();
    for(int iDim = 0; iDim < nDim; iDim++)
    {
      const auto& subT = TT.subTensor(iDim);
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

  // helper function to call leftNormalize on a tensor train and do common checks (with boundary rank support)
  void check_leftNormalize_boundaryRank(TensorTrain_double& TT)
  {
    const TensorTrain_double refTT = TT;
    const double TTnorm = leftNormalize(TT);

    EXPECT_NEAR(norm2(refTT), TTnorm, eps);
    EXPECT_NEAR(1., norm2(TT), eps);
    EXPECT_TRUE(static_cast<bool>(PITTS::TT_Orthogonality::left & TT.isOrthogonal()));

    // check orthogonality of subtensors
    const int nDim = TT.dimensions().size();
    for(int iDim = 0; iDim < nDim; iDim++)
    {
      const auto& subT = TT.subTensor(iDim);
      
      if( iDim != nDim-1 )
      {
        const mat orthogErr = ConstEigenMap(leftContract(subT)) - mat::Identity(subT.r2(),subT.r2());
        EXPECT_NEAR(0., orthogErr.norm(), eps);
      }
      else
      {
        const double normalizeErr = PITTS::internal::t3_nrm(subT) - 1;
        EXPECT_NEAR(0., normalizeErr, eps);
      }
    }

    // check tensor is the same, except for scaling...
    TensorTrain_double testTT(TT.dimensions());
    copy(TT, testTT);
    for(int i = 0; i < 10; i++)
    {
      randomize(testTT);
      EXPECT_NEAR(dot(refTT,testTT), TTnorm*dot(TT,testTT), eps*norm2(testTT));
    }
  }

  // helper function to call leftNormalize on a tensor train and do common checks (with boundary rank support)
  void check_rightNormalize_boundaryRank(TensorTrain_double& TT)
  {
    const TensorTrain_double refTT = TT;
    const double TTnorm = rightNormalize(TT);

    EXPECT_NEAR(norm2(refTT), TTnorm, eps);
    EXPECT_NEAR(1., norm2(TT), eps);
    EXPECT_TRUE(static_cast<bool>(PITTS::TT_Orthogonality::right & TT.isOrthogonal()));

    // check orthogonality of subtensors
    const int nDim = TT.dimensions().size();
    for(int iDim = 0; iDim < nDim; iDim++)
    {
      const auto& subT = TT.subTensor(iDim);
      
      if( iDim != 0 )
      {
        const mat orthogErr = ConstEigenMap(rightContract(subT)) - mat::Identity(subT.r1(),subT.r1());
        EXPECT_NEAR(0., orthogErr.norm(), eps);
      }
      else
      {
        const double normalizeErr = PITTS::internal::t3_nrm(subT) - 1;
        EXPECT_NEAR(0., normalizeErr, eps);
      }
    }

    // check tensor is the same, except for scaling...
    TensorTrain_double testTT(TT.dimensions());
    copy(TT, testTT);
    for(int i = 0; i < 10; i++)
    {
      randomize(testTT);
      EXPECT_NEAR(dot(refTT,testTT), TTnorm*dot(TT,testTT), eps*norm2(testTT));
    }
  }

  // helper function to determine the squared distance between two tensor trains: ||TTx - TTy||_2^2
  double squaredDistance(const TensorTrain_double& TTx, const TensorTrain_double& TTy, double scaleY)
  {
    // avoid axpby as it depends on normalize!
    // So use quadratic expansion ||x-y||_2^2 = <x,x> - 2<x,y> + <y,y>
    using PITTS::dot;
    return dot(TTx,TTx) - 2*scaleY*dot(TTx,TTy) + scaleY*scaleY*dot(TTy,TTy);
  }

  // helper function to return the maximal rank of a tensor train
  int maxRank(const TensorTrain_double& TT)
  {
    const auto& r = TT.getTTranks();
    return *std::max_element(std::begin(r), std::end(r));
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

  const int nDim = TT.dimensions().size();
  Tensor3_double t3;
  for(int iDim = 0; iDim < nDim; iDim++)
  {
    t3.resize(TT.subTensor(iDim).r1(), TT.subTensor(iDim).n(), TT.subTensor(iDim).r2());
    t3.setConstant(1.);
    t3 = TT.setSubTensor(iDim, std::move(t3));
  }

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

  const int nDim = TT.dimensions().size();
  Tensor3_double t3;
  for(int iDim = 0; iDim < nDim; iDim++)
  {
    t3.resize(TT.subTensor(iDim).r1(), TT.subTensor(iDim).n(), TT.subTensor(iDim).r2());
    t3.setConstant(1.);
    t3 = TT.setSubTensor(iDim, std::move(t3));
  }

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

  const int nDim = TT.dimensions().size();
  Tensor3_double t3;
  for(int iDim = 0; iDim < nDim; iDim++)
  {
    t3.resize(TT.subTensor(iDim).r1(), TT.subTensor(iDim).n(), TT.subTensor(iDim).r2());
    t3.setConstant(1.);
    t3 = TT.setSubTensor(iDim, std::move(t3));
  }

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

TEST(PITTS_TensorTrain_normalize, approximation_error_d2)
{
  TensorTrain_double TT(2,50);
  TT.setTTranks(50);

  // calculate random orthogonal matrices
  Eigen::MatrixXd Q1 = randomOrthoMatrix(50,50);
  Eigen::MatrixXd Q2 = randomOrthoMatrix(50,50);

  // desired singular value distribution
  Eigen::VectorXd sigma_sqrt(50);
  for(int i = 0; i < 50; i++)
    sigma_sqrt(i) = std::sqrt(1./(i+1));
  Eigen::MatrixXd t1 = Q1 * sigma_sqrt.asDiagonal();
  Eigen::MatrixXd t2 = sigma_sqrt.asDiagonal() * Q2;

  // Calculat the expected error when truncating at a specific singular value
  std::vector<double> squaredTruncationError(51);
  squaredTruncationError.back() = 0;
  for(int i = 50-1; i >= 0; i--)
    squaredTruncationError[i] = squaredTruncationError[i+1] + std::pow(sigma_sqrt(i),4);

  // set Tensor-Train cores
  Tensor3_double subT0(1, 50, 50);
  Tensor3_double subT1(50, 50, 1);
  for(int i = 0; i < 50; i++)
    for(int j = 0; j < 50; j++)
    {
      subT0(0,i,j) = t1(i,j);
      subT1(i,j,0) = t2(i,j);
    }
  TT.setSubTensor(0, std::move(subT0));
  TT.setSubTensor(1, std::move(subT1));

  // calculate the norm for checking relative errors
  double nrm_ref = PITTS::norm2(TT);

  // now try different truncations...
  TensorTrain_double TTtruncated(2,50);

  // full accuaracy
  PITTS::copy(TT, TTtruncated);
  double nrm = PITTS::normalize(TTtruncated);
  double squaredError = squaredDistance(TT, TTtruncated, nrm);
  EXPECT_NEAR(0., squaredError, eps);
  EXPECT_EQ(std::vector<int>({50}), TTtruncated.getTTranks());

  // reduce required accuracy
  PITTS::copy(TT, TTtruncated);
  nrm = PITTS::normalize(TTtruncated, 1/25.5);
  squaredError = squaredDistance(TT, TTtruncated, nrm);
  EXPECT_NEAR(squaredTruncationError[25], squaredError, eps);
  EXPECT_EQ(std::vector<int>({25}), TTtruncated.getTTranks());

  // further reduce required accuracy
  PITTS::copy(TT, TTtruncated);
  nrm = PITTS::normalize(TTtruncated, 1/10.5);
  squaredError = squaredDistance(TT, TTtruncated, nrm);
  EXPECT_NEAR(squaredTruncationError[10], squaredError, eps);
  EXPECT_EQ(std::vector<int>({10}), TTtruncated.getTTranks());


  // other variant: enforce maximal tensor dimension
  PITTS::copy(TT, TTtruncated);
  nrm = PITTS::normalize(TTtruncated, 1.e-10, 30);
  squaredError = squaredDistance(TT, TTtruncated, nrm);
  EXPECT_NEAR(squaredTruncationError[30], squaredError, eps);
  EXPECT_EQ(std::vector<int>({30}), TTtruncated.getTTranks());

  PITTS::copy(TT, TTtruncated);
  nrm = PITTS::normalize(TTtruncated, 1.e-10, 5);
  squaredError = squaredDistance(TT, TTtruncated, nrm);
  EXPECT_NEAR(squaredTruncationError[5], squaredError, eps);
  EXPECT_EQ(std::vector<int>({5}), TTtruncated.getTTranks());
}


TEST(PITTS_TensorTrain_normalize, approximation_error_d10)
{
  TensorTrain_double TT(10,2);
  // more difficult to make useful test case in higher dimensions, just make it random...
  TT.setTTranks(32);
  PITTS::randomize(TT);
  // workaround to orthogonalize it wrt. the tt core in the middle
  {
    TensorTrain_double tmp(10,2);
    PITTS::rightNormalize(TT);
    PITTS::copy(TT, tmp);
    PITTS::leftNormalize(tmp);
    {
      const auto& subT = TT.subTensor(5);
      Tensor3_double t3(subT.r1(), subT.n(), subT.r2());
      for(int i = 0; i < subT.r1(); i++)
        for(int j = 0; j < subT.n(); j++)
          for(int k = 0; k < subT.r2(); k++)
            t3(i,j,k) = subT(i,j,k) * std::pow(0.1, i);
      TT.setSubTensor(5, std::move(t3));
    }
    ASSERT_EQ(tmp.subTensor(4).r2(), TT.subTensor(5).r1());
    {
      std::vector<Tensor3_double> leftHalf(5);;
      for(int iDim = 0; iDim < 5; iDim++)
        copy(tmp.subTensor(iDim), leftHalf[iDim]);
      TT.setSubTensors(0, std::move(leftHalf));
    }
  }

  // try different truncation accuracies...
  TensorTrain_double TTtruncated(10,2);

  // default accuracy
  PITTS::copy(TT, TTtruncated);
  double nrm = PITTS::normalize(TTtruncated);
  double squaredError = squaredDistance(TT, TTtruncated, nrm);
  EXPECT_NEAR(0, squaredError, eps);

  // less accuracy
  PITTS::copy(TT, TTtruncated);
  nrm = PITTS::normalize(TTtruncated, 0.1);
  squaredError = squaredDistance(TT, TTtruncated, nrm);
  EXPECT_NEAR(0.05, std::sqrt(squaredError), 0.05);
  //std::cout << "error: " << std::sqrt(squaredError) << "\n";
  //std::cout << "TT-ranks:";
  //for(auto r: TTtruncated.getTTranks())
  //  std::cout << " " << r;
  //std::cout << "\n";
  EXPECT_LT(maxRank(TTtruncated), maxRank(TT));

  // some more accuracy
  PITTS::copy(TT, TTtruncated);
  nrm = PITTS::normalize(TTtruncated, 0.01);
  squaredError = squaredDistance(TT, TTtruncated, nrm);
  EXPECT_NEAR(0.005, std::sqrt(squaredError), 0.005);
  //std::cout << "error: " << std::sqrt(squaredError) << "\n";
  //std::cout << "TT-ranks:";
  //for(auto r: TTtruncated.getTTranks())
  //  std::cout << " " << r;
  //std::cout << "\n";
  EXPECT_LT(maxRank(TTtruncated), maxRank(TT));

  // and some more accuracy
  PITTS::copy(TT, TTtruncated);
  nrm = PITTS::normalize(TTtruncated, 0.001);
  squaredError = squaredDistance(TT, TTtruncated, nrm);
  EXPECT_NEAR(0.0005, std::sqrt(squaredError), 0.0005);
  //std::cout << "error: " << std::sqrt(squaredError) << "\n";
  //std::cout << "TT-ranks:";
  //for(auto r: TTtruncated.getTTranks())
  //  std::cout << " " << r;
  //std::cout << "\n";
  EXPECT_LT(maxRank(TTtruncated), maxRank(TT));
}

TEST(PITTS_TensorTrain_normalize, rightNormalize_same_as_leftNormalize_reversed)
{
  TensorTrain_double TT({10,9,8,7});
  TT.setTTranks({5,4,4});
  randomize(TT);

  constexpr auto transpose = [](const Tensor3_double& A, Tensor3_double& B)
  {
    B.resize(A.r2(),A.n(),A.r1());
    for(int i = 0; i < A.r1(); i++)
      for(int j = 0; j < A.n(); j++)
        for(int k = 0; k < A.r2(); k++)
          B(k,j,i) = A(i,j,k);
  };

  const auto nDim = TT.dimensions().size();
  std::vector<Tensor3_double> reversed_subT(nDim);
  for(int iDim = 0; iDim < nDim; iDim++)
    transpose(TT.subTensor(iDim), reversed_subT[nDim-1-iDim]);
  TensorTrain_double TT_reversed(std::move(reversed_subT));

  const double nrm = rightNormalize(TT, 0.1, 3);
  const double nrm_ref = leftNormalize(TT_reversed, 0.1, 3);
  EXPECT_NEAR(nrm_ref, nrm, eps);
  for(int iDim = 0; iDim < nDim; iDim++)
  {
    const auto& subT = TT.subTensor(iDim);
    const auto& subT_reversed = TT_reversed.subTensor(nDim-1-iDim);
    Tensor3_double subT_ref;
    transpose(subT_reversed, subT_ref);
    ASSERT_EQ(subT_ref.r1(), subT.r1());
    ASSERT_EQ(subT_ref.n(), subT.n());
    ASSERT_EQ(subT_ref.r2(), subT.r2());
    for(int i = 0; i < subT.r1(); i++)
      for(int j = 0; j < subT.n(); j++)
        for(int k = 0; k < subT.r2(); k++)
        {
          // sign of singular vectors is arbitrary, so just compare absolute values
          EXPECT_NEAR(std::abs(subT_ref(i,j,k)), std::abs(subT(i,j,k)), eps);
        }
  }
}

TEST(PITTS_TensorTrain_normalize, leftNormalize_boundaryRank_nDim1)
{
  using PITTS::internal::t3_nrm;

  TensorTrain_double TT(1,5);
  {
    Tensor3_double subT(7,5,3);
    subT.setConstant(1);
    TT.setSubTensor(0, std::move(subT));
  }

  double nrm = leftNormalize(TT);

  double nrm_ref = std::sqrt(7*5*3);
  EXPECT_NEAR(nrm_ref, nrm, eps);
  const auto& subT = TT.subTensor(0);
  EXPECT_NEAR(1., t3_nrm(subT), eps);
  for(int i = 0; i < subT.r1(); i++)
    for(int j = 0; j < subT.n(); j++)
      for(int k = 0; k < subT.r2(); k++)
      {
        EXPECT_NEAR(1./nrm_ref, subT(i,j,k), eps);
      }

  randomize(TT);
  nrm_ref = t3_nrm(TT.subTensor(0));

  nrm = leftNormalize(TT);
  EXPECT_NEAR(nrm_ref, nrm, eps);
  EXPECT_NEAR(1., t3_nrm(TT.subTensor(0)), eps);
}

TEST(PITTS_TensorTrain_normalize, rightNormalize_boundaryRank_nDim1)
{
  using PITTS::internal::t3_nrm;

  TensorTrain_double TT(1,5);
  {
    Tensor3_double subT(7,5,3);
    subT.setConstant(1);
    TT.setSubTensor(0, std::move(subT));
  }

  double nrm = rightNormalize(TT);

  double nrm_ref = std::sqrt(7*5*3);
  EXPECT_NEAR(nrm_ref, nrm, eps);
  const auto& subT = TT.subTensor(0);
  EXPECT_NEAR(1., t3_nrm(subT), eps);
  for(int i = 0; i < subT.r1(); i++)
    for(int j = 0; j < subT.n(); j++)
      for(int k = 0; k < subT.r2(); k++)
      {
        EXPECT_NEAR(1./nrm_ref, subT(i,j,k), eps);
      }

  randomize(TT);
  nrm_ref = t3_nrm(TT.subTensor(0));

  nrm = rightNormalize(TT);
  EXPECT_NEAR(nrm_ref, nrm, eps);
  EXPECT_NEAR(1., t3_nrm(subT), eps);
}

TEST(PITTS_TensorTrain_normalize, leftNormalize_boundaryRank_nDim1_random)
{
  TensorTrain_double TT(1,5);
  {
    Tensor3_double subT(7,5,3);
    randomize(subT);
    TT.setSubTensor(0, std::move(subT));
  }

  check_leftNormalize_boundaryRank(TT);
}

TEST(PITTS_TensorTrain_normalize, leftNormalize_boundaryRank_nDim4_random)
{
  TensorTrain_double TT({3,4,5,6});
  TT.setTTranks(2);
  {
    Tensor3_double subTl(7,3,2);
    Tensor3_double subTr(2,6,3);
    TT.setSubTensor(0, std::move(subTl));
    TT.setSubTensor(3, std::move(subTr));
  }
  randomize(TT);

  check_leftNormalize_boundaryRank(TT);
}

TEST(PITTS_TensorTrain_normalize, rightNormalize_boundaryRank_nDim1_random)
{
  TensorTrain_double TT(1,5);
  {
    Tensor3_double subT(7,5,3);
    randomize(subT);
    TT.setSubTensor(0, std::move(subT));
  }

  check_rightNormalize_boundaryRank(TT);
}

TEST(PITTS_TensorTrain_normalize, rightNormalize_boundaryRank_nDim4_random)
{
  TensorTrain_double TT({3,4,5,6});
  TT.setTTranks(2);
  {
    Tensor3_double subTl(7,3,2);
    Tensor3_double subTr(2,6,3);
    TT.setSubTensor(0, std::move(subTl));
    TT.setSubTensor(3, std::move(subTr));
  }
  randomize(TT);

  check_rightNormalize_boundaryRank(TT);
}

TEST(PITTS_TensorTrain_normalize, isOrthogonal)
{
  TensorTrain_double TT({3,3,3,3});
  TT.setOnes();
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthogonal());
  leftNormalize(TT);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthogonal());
  Tensor3_double subT(1,3,1);
  subT.setConstant(2.0);
  TT.setSubTensor(0, std::move(subT));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthogonal());
  rightNormalize(TT);
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthogonal());
  leftNormalize(TT);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthogonal());
}

TEST(PITTS_TensorTrain_normalize, leftOrtho_range)
{
  using PITTS::internal::ensureLeftOrtho_range;
  TensorTrain_double TT({3,3,3,3,3,3});
  TT.setOnes();
  ensureLeftOrtho_range(TT, 1, 3);
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(5));
  ensureLeftOrtho_range(TT, 0, 2);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(5));
  ensureLeftOrtho_range(TT, 0, 4);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(5));
}

TEST(PITTS_TensorTrain_normalize, rightOrtho_range)
{
  using PITTS::internal::ensureRightOrtho_range;
  TensorTrain_double TT({3,3,3,3,3,3});
  TT.setOnes();
  ensureRightOrtho_range(TT, 1, 3);
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(5));
  ensureRightOrtho_range(TT, 0, 2);
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(5));
  ensureRightOrtho_range(TT, 0, 4);
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(5));
}

TEST(PITTS_TensorTrain_normalize, sweep_with_leftRightOrtho_range)
{
  using PITTS::internal::ensureLeftOrtho_range;
  using PITTS::internal::ensureRightOrtho_range;
  TensorTrain_double TT({3,3,3,3,3,3});
  TT.setOnes();
  //ensureLeftOrtho_range(TT, 0, 0);
  ensureRightOrtho_range(TT, 1, 5);
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(5));

  ensureLeftOrtho_range(TT, 0, 1);
  ensureRightOrtho_range(TT, 2, 5);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(5));

  ensureLeftOrtho_range(TT, 0, 2);
  ensureRightOrtho_range(TT, 3, 5);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(5));

  ensureLeftOrtho_range(TT, 0, 3);
  ensureRightOrtho_range(TT, 4, 5);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(5));

  ensureLeftOrtho_range(TT, 0, 4);
  //ensureRightOrtho_range(TT, 5, 5);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(5));

  ensureLeftOrtho_range(TT, 0, 3);
  ensureRightOrtho_range(TT, 4, 5);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(5));

  ensureLeftOrtho_range(TT, 0, 2);
  ensureRightOrtho_range(TT, 3, 5);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(5));

  ensureLeftOrtho_range(TT, 0, 1);
  ensureRightOrtho_range(TT, 2, 5);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(5));

  ensureLeftOrtho_range(TT, 0, 1);
  ensureRightOrtho_range(TT, 1, 5);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TT.isOrthonormal(0));
  EXPECT_EQ(PITTS::TT_Orthogonality::none, TT.isOrthonormal(1));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(2));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(3));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(4));
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TT.isOrthonormal(5));
}
