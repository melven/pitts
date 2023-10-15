// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

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
  ASSERT_EQ(1, TT1.subTensor(0).r1());
  ASSERT_EQ(3, TT1.subTensor(0).n());
  ASSERT_EQ(1, TT1.subTensor(0).r2());
  ASSERT_THROW(TT1.subTensor(1), std::out_of_range);
}

TEST(PITTS_TensorTrain, create_n_d)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  TensorTrain_double TT1(3,7);
  ASSERT_EQ(std::vector<int>({7,7,7}), TT1.dimensions());
  ASSERT_EQ(std::vector<int>({1,1}), TT1.getTTranks());
  for(int i = 0; i < 3; i++)
  {
    const auto& subT = TT1.subTensor(i);
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(7, subT.n());
    ASSERT_EQ(1, subT.r2());
  }

  TensorTrain_double TT2(2,3,4);
  ASSERT_EQ(std::vector<int>({3,3}), TT2.dimensions());
  ASSERT_EQ(std::vector<int>({4}), TT2.getTTranks());
  ASSERT_EQ(1, TT2.subTensor(0).r1());
  ASSERT_EQ(3, TT2.subTensor(0).n());
  ASSERT_EQ(4, TT2.subTensor(0).r2());
  ASSERT_EQ(4, TT2.subTensor(1).r1());
  ASSERT_EQ(3, TT2.subTensor(1).n());
  ASSERT_EQ(1, TT2.subTensor(1).r2());
  ASSERT_THROW(TT2.subTensor(2), std::out_of_range);
}

TEST(PITTS_TensorTrain, create_generic)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  const std::vector<int> dims = {2,3,4,5,6};

  TensorTrain_double TT(dims, 3);
  ASSERT_EQ(dims, TT.dimensions());
  ASSERT_EQ(std::vector<int>({3,3,3,3}), TT.getTTranks());
  ASSERT_EQ(1, TT.subTensor(0).r1());
  ASSERT_EQ(2, TT.subTensor(0).n());
  ASSERT_EQ(3, TT.subTensor(0).r2());
  ASSERT_EQ(3, TT.subTensor(1).r1());
  ASSERT_EQ(3, TT.subTensor(1).n());
  ASSERT_EQ(3, TT.subTensor(1).r2());
  ASSERT_EQ(3, TT.subTensor(2).r1());
  ASSERT_EQ(4, TT.subTensor(2).n());
  ASSERT_EQ(3, TT.subTensor(2).r2());
  ASSERT_EQ(3, TT.subTensor(3).r1());
  ASSERT_EQ(5, TT.subTensor(3).n());
  ASSERT_EQ(3, TT.subTensor(3).r2());
  ASSERT_EQ(3, TT.subTensor(4).r1());
  ASSERT_EQ(6, TT.subTensor(4).n());
  ASSERT_EQ(1, TT.subTensor(4).r2());
  ASSERT_THROW(TT.subTensor(5), std::out_of_range);
}

TEST(PITTS_TensorTrain, create_fromSubTensors)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;

  // create some reference tensor train
  TensorTrain_double TTref({2, 3, 4, 5, 6});
  TTref.setTTranks({2, 3, 3, 2});
  for(int iDim = 0; iDim < TTref.dimensions().size(); iDim++)
  {
    const int r1 = TTref.subTensor(iDim).r1();
    const int n = TTref.subTensor(iDim).n();
    const int r2 = TTref.subTensor(iDim).r2();
    Tensor3_double subT(r1, n, r2);
    for(int i = 0; i < r1; i++)
      for(int j = 0; j < n; j++)
        for(int k = 0; k < r2; k++)
          subT(i,j,k) = iDim*1000 + i + j*10 + k*100;
    TTref.setSubTensor(iDim, std::move(subT));
  }

  // copy subTensors
  std::vector<Tensor3_double> subTensors(TTref.dimensions().size());
  for(int iDim = 0; iDim < TTref.dimensions().size(); iDim++)
  {
    const auto& subTref = TTref.subTensor(iDim);
    const int r1 = subTref.r1();
    const int n = subTref.n();
    const int r2 = subTref.r2();
    subTensors[iDim].resize(r1, n, r2);
    for(int i = 0; i < r1; i++)
      for(int j = 0; j < n; j++)
        for(int k = 0; k < r2; k++)
          subTensors[iDim](i,j,k) = subTref(i,j,k);
  }

  // generate new TT from it
  TensorTrain_double TT(std::move(subTensors));

  ASSERT_EQ(TTref.dimensions(), TT.dimensions());
  for(int iDim = 0; iDim < TTref.dimensions().size(); iDim++)
  {
    const auto& subTref = TTref.subTensor(iDim);
    const auto& subT = TT.subTensor(iDim);
    ASSERT_EQ(subTref.r1(), subT.r1());
    ASSERT_EQ(subTref.n(), subT.n());
    ASSERT_EQ(subTref.r2(), subT.r2());
    for(int i = 0; i < subT.r1(); i++)
      for(int j = 0; j < subT.n(); j++)
        for(int k = 0; k < subT.r2(); k++)
        {
          ASSERT_EQ(subTref(i,j,k), subT(i,j,k));
        }
  }
}

TEST(PITTS_TensorTrain, setSubTensor)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;

  const std::vector<int> dims = {2,5,3};
  TensorTrain_double TT(dims, 3);

  ASSERT_EQ(3, TT.subTensor(1).r1());
  ASSERT_EQ(5, TT.subTensor(1).n());
  ASSERT_EQ(3, TT.subTensor(1).r2());

  Tensor3_double newSubT(3, 5, 3);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 3; k++)
        newSubT(i,j,k) = i + 100*j + 10000*k;

  // store pointer addresses
  const double* oldSubT_address = &TT.subTensor(1)(0,0,0);
  const double* newSubT_address = &newSubT(0,0,0);

  Tensor3_double oldSubT = TT.setSubTensor(1, std::move(newSubT));

  // check pointer addresses
  ASSERT_EQ(newSubT_address, &TT.subTensor(1)(0,0,0));
  ASSERT_EQ(oldSubT_address, &oldSubT(0,0,0));

  ASSERT_EQ(3, TT.subTensor(1).r1());
  ASSERT_EQ(5, TT.subTensor(1).n());
  ASSERT_EQ(3, TT.subTensor(1).r2());
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 3; k++)
      {
        EXPECT_EQ(i + 100*j + 10000*k, TT.subTensor(1)(i,j,k));
      }
}

TEST(PITTS_TensorTrain, setSubTensors_vector)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;

  const std::vector<int> dims = {2,5,3};
  TensorTrain_double TT(dims, 3);

  ASSERT_EQ(3, TT.subTensor(1).r1());
  ASSERT_EQ(5, TT.subTensor(1).n());
  ASSERT_EQ(3, TT.subTensor(1).r2());

  ASSERT_EQ(3, TT.subTensor(2).r1());
  ASSERT_EQ(3, TT.subTensor(2).n());
  ASSERT_EQ(1, TT.subTensor(2).r2());

  std::vector<Tensor3_double> newSubT(2);
  newSubT[0].resize(3, 5, 2);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 2; k++)
        newSubT[0](i,j,k) = i + 100*j + 10000*k;
  newSubT[1].resize(2, 3, 4);
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 4; k++)
        newSubT[1](i,j,k) = - (i + 100*j + 10000*k);

  // store pointer addresses
  const double* oldSubT1_address = &TT.subTensor(1)(0,0,0);
  const double* newSubT0_address = &newSubT[0](0,0,0);
  const double* oldSubT2_address = &TT.subTensor(2)(0,0,0);
  const double* newSubT1_address = &newSubT[1](0,0,0);

  std::vector<Tensor3_double> oldSubT = TT.setSubTensors(1, std::move(newSubT));

  // check pointer addresses
  ASSERT_EQ(newSubT0_address, &TT.subTensor(1)(0,0,0));
  ASSERT_EQ(newSubT1_address, &TT.subTensor(2)(0,0,0));
  ASSERT_EQ(2, oldSubT.size());
  ASSERT_EQ(oldSubT1_address, &oldSubT[0](0,0,0));
  ASSERT_EQ(oldSubT2_address, &oldSubT[1](0,0,0));

  ASSERT_EQ(3, TT.subTensor(1).r1());
  ASSERT_EQ(5, TT.subTensor(1).n());
  ASSERT_EQ(2, TT.subTensor(1).r2());
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 2; k++)
      {
        EXPECT_EQ(i + 100*j + 10000*k, TT.subTensor(1)(i,j,k));
      }

  ASSERT_EQ(2, TT.subTensor(2).r1());
  ASSERT_EQ(3, TT.subTensor(2).n());
  ASSERT_EQ(4, TT.subTensor(2).r2());
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 4; k++)
      {
        EXPECT_EQ(-(i + 100*j + 10000*k), TT.subTensor(2)(i,j,k));
      }
}

TEST(PITTS_TensorTrain, setSubTensors_vector_invalidDims)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;

  const std::vector<int> dims = {2,5,3};
  TensorTrain_double TT(dims, 3);

  std::vector<Tensor3_double> newSubT(2);

  // wrong dimension
  newSubT[0].resize(3, 5, 2);
  newSubT[1].resize(2, 4, 1);
  EXPECT_THROW(TT.setSubTensors(1, std::move(newSubT)), std::invalid_argument);

  // wrong rank
  newSubT[0].resize(2, 5, 2);
  newSubT[1].resize(2, 3, 1);
  EXPECT_THROW(TT.setSubTensors(1, std::move(newSubT)), std::invalid_argument);

  // inconsistent inner rank
  newSubT[0].resize(3, 5, 7);
  newSubT[1].resize(2, 3, 1);
  EXPECT_THROW(TT.setSubTensors(1, std::move(newSubT)), std::invalid_argument);
}

TEST(PITTS_TensorTrain, setSubTensors_otherTT)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;

  const std::vector<int> dims = {2,5,3};
  TensorTrain_double TT(dims, 3);

  ASSERT_EQ(3, TT.subTensor(1).r1());
  ASSERT_EQ(5, TT.subTensor(1).n());
  ASSERT_EQ(3, TT.subTensor(1).r2());

  ASSERT_EQ(3, TT.subTensor(2).r1());
  ASSERT_EQ(3, TT.subTensor(2).n());
  ASSERT_EQ(1, TT.subTensor(2).r2());

  std::vector<Tensor3_double> newSubT(2);
  newSubT[0].resize(3, 5, 2);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 2; k++)
        newSubT[0](i,j,k) = i + 100*j + 10000*k;
  newSubT[1].resize(2, 3, 4);
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 4; k++)
        newSubT[1](i,j,k) = - (i + 100*j + 10000*k);
  TensorTrain_double otherTT(std::move(newSubT));

  // store pointer addresses
  const double* oldSubT1_address = &TT.subTensor(1)(0,0,0);
  const double* newSubT0_address = &otherTT.subTensor(0)(0,0,0);
  const double* oldSubT2_address = &TT.subTensor(2)(0,0,0);
  const double* newSubT1_address = &otherTT.subTensor(1)(0,0,0);

  TensorTrain_double oldTT = TT.setSubTensors(1, std::move(otherTT));

  // check pointer addresses
  ASSERT_EQ(newSubT0_address, &TT.subTensor(1)(0,0,0));
  ASSERT_EQ(newSubT1_address, &TT.subTensor(2)(0,0,0));
  ASSERT_EQ(oldSubT1_address, &oldTT.subTensor(0)(0,0,0));
  ASSERT_EQ(oldSubT2_address, &oldTT.subTensor(1)(0,0,0));

  ASSERT_EQ(3, TT.subTensor(1).r1());
  ASSERT_EQ(5, TT.subTensor(1).n());
  ASSERT_EQ(2, TT.subTensor(1).r2());
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 2; k++)
      {
        EXPECT_EQ(i + 100*j + 10000*k, TT.subTensor(1)(i,j,k));
      }

  ASSERT_EQ(2, TT.subTensor(2).r1());
  ASSERT_EQ(3, TT.subTensor(2).n());
  ASSERT_EQ(4, TT.subTensor(2).r2());
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 4; k++)
      {
        EXPECT_EQ(-(i + 100*j + 10000*k), TT.subTensor(2)(i,j,k));
      }
}


TEST(PITTS_TensorTrain, setTTranks)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  TensorTrain_double TT(3,7);
  ASSERT_EQ(std::vector<int>({1,1}), TT.getTTranks());

  TT.setTTranks(2);
  ASSERT_EQ(std::vector<int>({2,2}), TT.getTTranks());
  ASSERT_EQ(1, TT.subTensor(0).r1());
  ASSERT_EQ(7, TT.subTensor(0).n());
  ASSERT_EQ(2, TT.subTensor(0).r2());
  ASSERT_EQ(2, TT.subTensor(1).r1());
  ASSERT_EQ(7, TT.subTensor(1).n());
  ASSERT_EQ(2, TT.subTensor(1).r2());
  ASSERT_EQ(2, TT.subTensor(2).r1());
  ASSERT_EQ(7, TT.subTensor(2).n());
  ASSERT_EQ(1, TT.subTensor(2).r2());
  ASSERT_THROW(TT.subTensor(3), std::out_of_range);

  TT.setTTranks({3,4});
  ASSERT_EQ(std::vector<int>({3,4}), TT.getTTranks());
  ASSERT_EQ(1, TT.subTensor(0).r1());
  ASSERT_EQ(7, TT.subTensor(0).n());
  ASSERT_EQ(3, TT.subTensor(0).r2());
  ASSERT_EQ(3, TT.subTensor(1).r1());
  ASSERT_EQ(7, TT.subTensor(1).n());
  ASSERT_EQ(4, TT.subTensor(1).r2());
  ASSERT_EQ(4, TT.subTensor(2).r1());
  ASSERT_EQ(7, TT.subTensor(2).n());
  ASSERT_EQ(1, TT.subTensor(2).r2());
  ASSERT_THROW(TT.subTensor(3), std::out_of_range);
}

TEST(PITTS_TensorTrain, setZero)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  TensorTrain_double TT(2,4,5);
  TT.setZero();
  // check result is zero everywhere!
  ASSERT_EQ(std::vector<int>({4,4}), TT.dimensions());
  for(int i = 0; i < 2; i++)
  {
    const auto& subT = TT.subTensor(i);
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
  // check result is one everywhere!
  ASSERT_EQ(std::vector<int>({4,4}), TT.dimensions());
  for(int i = 0; i < 2; i++)
  {
    const auto& subT = TT.subTensor(i);
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
  // check result is a unit tensor everywhere!
  ASSERT_EQ(std::vector<int>({4,4,4}), TT.dimensions());
  for(int i = 0; i < 3; i++)
  {
    const auto& subT = TT.subTensor(i);
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
  using Tensor3_double = PITTS::Tensor3<double>;

  TensorTrain_double TT1(3,4,5);
  int dummyVal = 0;
  ASSERT_EQ(std::vector<int>({4,4,4}), TT1.dimensions());
  for(int i = 0; i < 3; i++)
  {
    Tensor3_double subT(TT1.subTensor(i).r1(), TT1.subTensor(i).n(), TT1.subTensor(i).r2());
    for(int i = 0; i < subT.r1(); i++)
      for(int j = 0; j < subT.n(); j++)
        for(int k = 0; k < subT.r2(); k++)
          subT(i,j,k) = dummyVal++;
    TT1.setSubTensor(i, std::move(subT));
  }

  const TensorTrain_double& TT1_const(TT1);
  TensorTrain_double TT2(3,4);
  copy(TT1_const, TT2);

  ASSERT_EQ(TT1.dimensions(), TT2.dimensions());
  for(int iDim = 0; iDim < TT1.dimensions().size(); iDim++)
  {
    const auto& subT1 = TT1.subTensor(iDim);
    const auto& subT2 = TT2.subTensor(iDim);
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
