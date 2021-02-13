#include <gtest/gtest.h>
#include "pitts_tensortrain.hpp"
#include <type_traits>

TEST(PITTS_TensorTrain, type_traits)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  // implicit copying is not desired
  ASSERT_TRUE(std::is_copy_constructible<TensorTrain_double>());
  ASSERT_FALSE(std::is_copy_assignable<TensorTrain_double>());

  // move / swap is ok
  ASSERT_TRUE(std::is_nothrow_move_constructible<TensorTrain_double>());
  ASSERT_TRUE(std::is_nothrow_move_assignable<TensorTrain_double>());
  ASSERT_TRUE(std::is_nothrow_swappable<TensorTrain_double>());
}

TEST(PITTS_TensorTrain, create_n_1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  TensorTrain_double TT1(1,3);
  ASSERT_EQ(std::vector<int>({3}), TT1.dimensions());
  ASSERT_EQ(std::vector<int>({}), TT1.getTTranks());
  ASSERT_EQ(1, TT1.subTensors().size());
  ASSERT_EQ(1, TT1.subTensors()[0].r1());
  ASSERT_EQ(3, TT1.subTensors()[0].n());
  ASSERT_EQ(1, TT1.subTensors()[0].r2());
}

TEST(PITTS_TensorTrain, create_n_d)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  TensorTrain_double TT1(3,7);
  ASSERT_EQ(std::vector<int>({7,7,7}), TT1.dimensions());
  ASSERT_EQ(std::vector<int>({1,1}), TT1.getTTranks());
  ASSERT_EQ(3, TT1.subTensors().size());
  for(const auto& subT: TT1.subTensors())
  {
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(7, subT.n());
    ASSERT_EQ(1, subT.r2());
  }

  TensorTrain_double TT2(2,3,4);
  ASSERT_EQ(std::vector<int>({3,3}), TT2.dimensions());
  ASSERT_EQ(std::vector<int>({4}), TT2.getTTranks());
  ASSERT_EQ(2, TT2.subTensors().size());
  ASSERT_EQ(1, TT2.subTensors()[0].r1());
  ASSERT_EQ(3, TT2.subTensors()[0].n());
  ASSERT_EQ(4, TT2.subTensors()[0].r2());
  ASSERT_EQ(4, TT2.subTensors()[1].r1());
  ASSERT_EQ(3, TT2.subTensors()[1].n());
  ASSERT_EQ(1, TT2.subTensors()[1].r2());
}

TEST(PITTS_TensorTrain, create_generic)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  const std::vector<int> dims = {2,3,4,5,6};

  TensorTrain_double TT(dims, 3);
  ASSERT_EQ(dims, TT.dimensions());
  ASSERT_EQ(std::vector<int>({3,3,3,3}), TT.getTTranks());
  ASSERT_EQ(5, TT.subTensors().size());
  ASSERT_EQ(1, TT.subTensors()[0].r1());
  ASSERT_EQ(2, TT.subTensors()[0].n());
  ASSERT_EQ(3, TT.subTensors()[0].r2());
  ASSERT_EQ(3, TT.subTensors()[1].r1());
  ASSERT_EQ(3, TT.subTensors()[1].n());
  ASSERT_EQ(3, TT.subTensors()[1].r2());
  ASSERT_EQ(3, TT.subTensors()[2].r1());
  ASSERT_EQ(4, TT.subTensors()[2].n());
  ASSERT_EQ(3, TT.subTensors()[2].r2());
  ASSERT_EQ(3, TT.subTensors()[3].r1());
  ASSERT_EQ(5, TT.subTensors()[3].n());
  ASSERT_EQ(3, TT.subTensors()[3].r2());
  ASSERT_EQ(3, TT.subTensors()[4].r1());
  ASSERT_EQ(6, TT.subTensors()[4].n());
  ASSERT_EQ(1, TT.subTensors()[4].r2());
}

TEST(PITTS_TensorTrain, setTTranks)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  TensorTrain_double TT(3,7);
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

TEST(PITTS_TensorTrain, setZero)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  TensorTrain_double TT(2,4,5);
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

TEST(PITTS_TensorTrain, setOnes)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  TensorTrain_double TT(2,4,5);
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

TEST(PITTS_TensorTrain, setUnit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  TensorTrain_double TT(3,4,5);
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

TEST(PITTS_TensorTrain, copy)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  TensorTrain_double TT1(3,4,5);
  int dummyVal = 0;
  for(auto& subT: TT1.editableSubTensors())
  {
    for(int i = 0; i < subT.r1(); i++)
      for(int j = 0; j < subT.n(); j++)
        for(int k = 0; k < subT.r2(); k++)
          subT(i,j,k) = dummyVal++;
  }

  const TensorTrain_double& TT1_const(TT1);
  TensorTrain_double TT2(3,4);
  copy(TT1_const, TT2);

  ASSERT_EQ(TT1.subTensors().size(), TT2.subTensors().size());
  for(int iDim = 0; iDim < TT1.subTensors().size(); iDim++)
  {
    const auto& subT1 = TT1.subTensors()[iDim];
    const auto& subT2 = TT2.subTensors()[iDim];
    ASSERT_EQ(subT1.r1(), subT2.r1());
    ASSERT_EQ(subT1.n(), subT2.n());
    ASSERT_EQ(subT1.r2(), subT2.r2());
    for(int i = 0; i < subT1.r1(); i++)
      for(int j = 0; j < subT1.n(); j++)
        for(int k = 0; k < subT1.r2(); k++)
        {
          ASSERT_EQ(subT1(i,j,k), subT2(i,j,k));
        }
  }
}
