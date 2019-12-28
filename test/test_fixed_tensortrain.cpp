#include <gtest/gtest.h>
#include "pitts_fixed_tensortrain.hpp"

TEST(PITTS_FixedTensorTrain, create_n_1)
{
  using FixedTensorTrain_double = PITTS::FixedTensorTrain<double,3>;

  FixedTensorTrain_double TT1(1);
  ASSERT_EQ(std::vector<int>({}), TT1.getTTranks());
  ASSERT_EQ(1, TT1.subTensors().size());
  ASSERT_EQ(1, TT1.subTensors()[0].r1());
  ASSERT_EQ(3, TT1.subTensors()[0].n());
  ASSERT_EQ(1, TT1.subTensors()[0].r2());
}

TEST(PITTS_FixedTensorTrain, create_n_d)
{
  using FixedTensorTrain7_double = PITTS::FixedTensorTrain<double,7>;
  using FixedTensorTrain3_double = PITTS::FixedTensorTrain<double,3>;

  FixedTensorTrain7_double TT1(3);
  ASSERT_EQ(std::vector<int>({1,1}), TT1.getTTranks());
  ASSERT_EQ(3, TT1.subTensors().size());
  for(const auto& subT: TT1.subTensors())
  {
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(7, subT.n());
    ASSERT_EQ(1, subT.r2());
  }

  FixedTensorTrain3_double TT2(2,4);
  ASSERT_EQ(std::vector<int>({4}), TT2.getTTranks());
  ASSERT_EQ(2, TT2.subTensors().size());
  ASSERT_EQ(1, TT2.subTensors()[0].r1());
  ASSERT_EQ(3, TT2.subTensors()[0].n());
  ASSERT_EQ(4, TT2.subTensors()[0].r2());
  ASSERT_EQ(4, TT2.subTensors()[1].r1());
  ASSERT_EQ(3, TT2.subTensors()[1].n());
  ASSERT_EQ(1, TT2.subTensors()[1].r2());
}

TEST(PITTS_FixedTensorTrain, setTTranks)
{
  using FixedTensorTrain_double = PITTS::FixedTensorTrain<double,7>;

  FixedTensorTrain_double TT(3);
  ASSERT_EQ(std::vector<int>({1,1}), TT.getTTranks());

  TT.setTTranks(2);
  ASSERT_EQ(std::vector<int>({2,2}), TT.getTTranks());
  ASSERT_EQ(3, TT.subTensors().size());
  ASSERT_EQ(1, TT.subTensors()[0].r1());
  ASSERT_EQ(7, TT.subTensors()[0].n());
  ASSERT_EQ(2, TT.subTensors()[0].r2());
  ASSERT_EQ(2, TT.subTensors()[1].r1());
  ASSERT_EQ(7, TT.subTensors()[1].n());
  ASSERT_EQ(2, TT.subTensors()[1].r2());
  ASSERT_EQ(2, TT.subTensors()[2].r1());
  ASSERT_EQ(7, TT.subTensors()[2].n());
  ASSERT_EQ(1, TT.subTensors()[2].r2());

  TT.setTTranks({3,4});
  ASSERT_EQ(std::vector<int>({3,4}), TT.getTTranks());
  ASSERT_EQ(3, TT.subTensors().size());
  ASSERT_EQ(1, TT.subTensors()[0].r1());
  ASSERT_EQ(7, TT.subTensors()[0].n());
  ASSERT_EQ(3, TT.subTensors()[0].r2());
  ASSERT_EQ(3, TT.subTensors()[1].r1());
  ASSERT_EQ(7, TT.subTensors()[1].n());
  ASSERT_EQ(4, TT.subTensors()[1].r2());
  ASSERT_EQ(4, TT.subTensors()[2].r1());
  ASSERT_EQ(7, TT.subTensors()[2].n());
  ASSERT_EQ(1, TT.subTensors()[2].r2());
}

TEST(PITTS_FixedTensorTrain, setZero)
{
  using FixedTensorTrain_double = PITTS::FixedTensorTrain<double,4>;

  FixedTensorTrain_double TT(2,5);
  TT.setZero();
  // check result is zero everywhere!
  for(const auto& subT: TT.subTensors())
  {
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(4, subT.n());
    ASSERT_EQ(1, subT.r2());
    for(int i = 0; i < 4; i++)
      ASSERT_EQ(0, subT(0,i,0));
  }
}

TEST(PITTS_FixedTensorTrain, setOnes)
{
  using FixedTensorTrain_double = PITTS::FixedTensorTrain<double,4>;

  FixedTensorTrain_double TT(2,5);
  TT.setOnes();
  // check result is zero everywhere!
  for(const auto& subT: TT.subTensors())
  {
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(4, subT.n());
    ASSERT_EQ(1, subT.r2());
    for(int i = 0; i < 4; i++)
      ASSERT_EQ(1, subT(0,i,0));
  }
}

TEST(PITTS_FixedTensorTrain, setUnit)
{
  using FixedTensorTrain_double = PITTS::FixedTensorTrain<double,4>;

  FixedTensorTrain_double TT(3,5);
  const std::vector<int> unitIdx = {0,3,1};
  TT.setUnit(unitIdx);
  // check result is zero everywhere!
  for(int i = 0; i < 3; i++)
  {
    const auto& subT = TT.subTensors()[i];
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(4, subT.n());
    ASSERT_EQ(1, subT.r2());
    for(int j = 0; j < 4; j++)
    {
      double ref = unitIdx[i] == j ? 1 : 0;
      ASSERT_EQ(ref, subT(0,j,0));
    }
  }
}

