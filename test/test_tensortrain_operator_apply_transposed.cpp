// Copyright (c) 2021 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_operator_apply_transposed.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_laplace_operator.hpp"

TEST(PITTS_TensorTrainOperator_apply_transposed, zero)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp(3, 5, 4);
  TTOp.setZero();

  TensorTrain_double TTx(3, 5, 7), TTy(3, 4);
  randomize(TTx);

  applyT(TTOp, TTx, TTy);

  ASSERT_EQ(TTx.getTTranks(), TTy.getTTranks());
  const int nDim = TTy.dimensions().size();
  for(int iDim = 0; iDim < nDim; iDim++)
  {
    const auto& subTy = TTy.subTensor(iDim);
    for(int i = 0; i < subTy.r1(); i++)
      for(int j = 0; j < subTy.n(); j++)
        for(int k = 0; k < subTy.r2(); k++)
          ASSERT_NEAR(0., subTy(i,j,k), eps);
  }
}

TEST(PITTS_TensorTrainOperator_apply_transposed, eye)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp(3, 5, 4);
  TTOp.setEye();

  TensorTrain_double TTx(3, 5, 7), TTy(3, 4);
  randomize(TTx);

  applyT(TTOp, TTx, TTy);

  ASSERT_EQ(TTx.getTTranks(), TTy.getTTranks());
  for(int iDim = 0; iDim < TTx.dimensions().size(); iDim++)
  {
    const auto& subTx = TTx.subTensor(iDim);
    const auto& subTy = TTy.subTensor(iDim);
    for(int i = 0; i < subTy.r1(); i++)
      for(int j = 0; j < subTy.n(); j++)
        for(int k = 0; k < subTy.r2(); k++)
          ASSERT_NEAR(subTx(i,j,k), subTy(i,j,k), eps);
  }
}

TEST(PITTS_TensorTrainOperator_apply_transposed, eye_axpy_eye)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp1(3, 5, 5);
  TensorTrainOperator_double TTOp2(3, 5, 5);
  TTOp1.setEye();
  TTOp2.setEye();
  axpby(1., TTOp1, 3., TTOp2);

  TensorTrain_double TTx(3, 5, 7), TTy(3, 5);
  randomize(TTx);

  applyT(TTOp2, TTx, TTy);

  ASSERT_EQ(TTx.getTTranks(), TTy.getTTranks());
  // TTy should be 4*TTx now
  const auto gamma = axpby(-4., TTx, 1., TTy);
  ASSERT_NEAR(0., gamma, eps);
}

TEST(PITTS_TensorTrainOperator_apply_transposed, laplace_operator)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOpLaplace(4, 5, 5);
  TTOpLaplace.setZero();
  TensorTrainOperator_double TTOpDummy(4, 5, 5);
  TTOpDummy.setEye();
  Tensor3_double tridi(1,5*5,1);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 5; j++)
    {
      if( i == j )
        tridi(0,i*5+j,0) = -2. / (5+1);
      else if( i == j+1 || i == j-1 )
        tridi(0,i*5+j,0) = 1. / (5+1);
      else
        tridi(0,i*5+j,0) = 0;
    }
  for(int iDim = 0; iDim < 4; iDim++)
  {
    Tensor3_double subT = TTOpDummy.tensorTrain().setSubTensor(iDim, std::move(tridi));
    axpby(1., TTOpDummy, 1., TTOpLaplace);
    tridi = TTOpDummy.tensorTrain().setSubTensor(iDim, std::move(subT));
  }


  TensorTrain_double TTx(4, 5, 3), TTy(4, 5), TTy_ref(4, 5);
  randomize(TTx);

  applyT(TTOpLaplace, TTx, TTy);
  copy(TTx, TTy_ref);
  const auto nrm_ref = laplaceOperator(TTy_ref);
  const auto error = axpby(-nrm_ref, TTy_ref, 1., TTy);
  EXPECT_NEAR(0., error, eps);
}

