#include <gtest/gtest.h>
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_laplace_operator.hpp"

TEST(PITTS_TensorTrainOperator_apply, zero)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp(3, 4, 5);
  TTOp.setZero();

  TensorTrain_double TTx(3, 5, 7), TTy(3, 4);
  randomize(TTx);

  apply(TTOp, TTx, TTy);

  ASSERT_EQ(TTx.getTTranks(), TTy.getTTranks());
  for(const auto& subTy: TTy.subTensors())
  {
    for(int i = 0; i < subTy.r1(); i++)
      for(int j = 0; j < subTy.n(); j++)
        for(int k = 0; k < subTy.r2(); k++)
          ASSERT_NEAR(0., subTy(i,j,k), eps);
  }
}

TEST(PITTS_TensorTrainOperator_apply, eye)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp(3, 4, 5);
  TTOp.setEye();

  TensorTrain_double TTx(3, 5, 7), TTy(3, 4);
  randomize(TTx);

  apply(TTOp, TTx, TTy);

  ASSERT_EQ(TTx.getTTranks(), TTy.getTTranks());
  for(int iDim = 0; iDim < TTx.subTensors().size(); iDim++)
  {
    const auto& subTx = TTx.subTensors()[iDim];
    const auto& subTy = TTy.subTensors()[iDim];
    for(int i = 0; i < subTy.r1(); i++)
      for(int j = 0; j < subTy.n(); j++)
        for(int k = 0; k < subTy.r2(); k++)
          ASSERT_NEAR(subTx(i,j,k), subTy(i,j,k), eps);
  }
}

TEST(PITTS_TensorTrainOperator_apply, eye_axpy_eye)
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

  apply(TTOp2, TTx, TTy);

  ASSERT_EQ(TTx.getTTranks(), TTy.getTTranks());
  // TTy should be 4*TTx now
  const auto gamma = axpby(-4., TTx, 1., TTy);
  ASSERT_NEAR(0., gamma, eps);
}

TEST(PITTS_TensorTrainOperator_apply, laplace_operator)
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
    auto& subT = TTOpDummy.tensorTrain().editableSubTensors()[iDim];
    std::swap(subT, tridi);
    axpby(1., TTOpDummy, 1., TTOpLaplace);
    std::swap(subT, tridi);
  }


  TensorTrain_double TTx(4, 5, 3), TTy(4, 5), TTy_ref(4, 5);
  randomize(TTx);

  apply(TTOpLaplace, TTx, TTy);
  copy(TTx, TTy_ref);
  const auto nrm_ref = laplaceOperator(TTy_ref);
  const auto error = axpby(-nrm_ref, TTy_ref, 1., TTy);
  EXPECT_NEAR(0., error, eps);
}

