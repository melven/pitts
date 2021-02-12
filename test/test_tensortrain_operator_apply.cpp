#include <gtest/gtest.h>
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_axpby.hpp"

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
