#include <gtest/gtest.h>
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_random.hpp"

TEST(PITTS_TensorTrain_norm, rank_1_vector)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(1,5);

  TT.setZero();
  EXPECT_EQ(0, norm2(TT));

  TT.setOnes();
  EXPECT_NEAR(std::sqrt(5.), norm2(TT), eps);

  TT.setUnit({2});
  EXPECT_NEAR(1, norm2(TT), eps);

  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);
}

TEST(PITTS_TensorTrain_norm, rank_2_matrix)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(2,5);

  TT.setZero();
  EXPECT_EQ(0, norm2(TT));

  TT.setOnes();
  EXPECT_NEAR(std::sqrt(25.), norm2(TT), eps);

  TT.setUnit({2,1});
  EXPECT_NEAR(1, norm2(TT), eps);

  // TT-ranks==1
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);

  // higher rank
  TT.setTTranks({3});
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);
}

TEST(PITTS_TensorTrain_norm, rank_4_tensor)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(4,2);

  TT.setZero();
  EXPECT_EQ(0, norm2(TT));

  TT.setOnes();
  EXPECT_NEAR(std::sqrt(2*2*2*2), norm2(TT), eps);

  TT.setUnit({0,1,1,0});
  EXPECT_NEAR(1, norm2(TT), eps);

  // TT-ranks==1
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);

  // higher TT-ranks
  TT.setTTranks({3,1,2});
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);
}

TEST(PITTS_TensorTrain_norm, large_rank_4_tensor)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(4,100);

  TT.setZero();
  EXPECT_EQ(0, norm2(TT));

  TT.setOnes();
  EXPECT_NEAR(std::sqrt(100*100*100*100), norm2(TT), eps);

  TT.setUnit({0,1,1,0});
  EXPECT_NEAR(1, norm2(TT), eps);

  // TT-ranks==1
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);

  // higher TT-ranks
  TT.setTTranks({3,1,2});
  randomize(TT);
  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);
}

TEST(PITTS_TensorTrain_norm, boundary_rank_nDim1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(1, 5);

  Tensor3_double subT(2,5,3);
  subT.setConstant(1);
  subT = TT.setSubTensor(0, std::move(subT));

  EXPECT_NEAR(std::sqrt(2*5*3), norm2(TT), eps);

  subT.resize(2,5,3);
  randomize(subT);
  subT = TT.setSubTensor(0, std::move(subT));
  copy(TT.subTensor(0), subT);

  double nrm_ref = 0;
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 3; k++)
        nrm_ref += subT(i,j,k)*subT(i,j,k);
  nrm_ref = std::sqrt(nrm_ref);

  EXPECT_NEAR(nrm_ref, norm2(TT), eps);
}

TEST(PITTS_TensorTrain_norm, boundary_rank_nDim2)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(2, 5);

  Tensor3_double subT1(2,5,3);
  Tensor3_double subT2(3,5,4);

  randomize(subT1);
  randomize(subT2);

  TT.setTTranks(3);
  TT.setSubTensor(0, std::move(subT1));
  TT.setSubTensor(1, std::move(subT2));

  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);
}

TEST(PITTS_TensorTrain_norm, boundary_rank_nDim6)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT({3,4,3,3,2,3});
  TT.setTTranks(2);
  randomize(TT);

  Tensor3_double subTl(3,3,2);
  Tensor3_double subTr(2,3,4);

  randomize(subTl);
  randomize(subTr);

  TT.setSubTensor(0, std::move(subTl));
  TT.setSubTensor(5, std::move(subTr));

  EXPECT_NEAR(std::sqrt(dot(TT,TT)), norm2(TT), eps);
}
