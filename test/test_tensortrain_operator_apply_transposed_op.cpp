#include <gtest/gtest.h>
#include "pitts_tensortrain_operator_apply_transposed_op.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_operator_apply_transposed.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_laplace_operator.hpp"

TEST(PITTS_TensorTrainOperator_apply_transposed_op, zero)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOpA(3, 5, 4);
  TTOpA.setZero();

  TensorTrainOperator_double TTOpB(3, 5, 2);
  TTOpB.setZero();

  TensorTrainOperator_double TTOpC(3, 4, 2);

  applyT(TTOpA, TTOpB, TTOpC);

  ASSERT_EQ(TTOpA.getTTranks(), TTOpC.getTTranks());
  const int nDim = TTOpC.tensorTrain().dimensions().size();
  for(int iDim = 0; iDim < nDim; iDim++)
  {
    const auto& subTOpC = TTOpC.tensorTrain().subTensor(iDim);
    for(int i = 0; i < subTOpC.r1(); i++)
      for(int j = 0; j < subTOpC.n(); j++)
        for(int k = 0; k < subTOpC.r2(); k++)
          ASSERT_NEAR(0., subTOpC(i,j,k), eps);
  }
}

TEST(PITTS_TensorTrainOperator_apply_transposed_op, eyeA)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOpA(3, 5, 4);
  TTOpA.setEye();

  TensorTrainOperator_double TTOpB(3, 5, 7), TTOpC(3, 4, 7);
  randomize(TTOpB);

  applyT(TTOpA, TTOpB, TTOpC);

  ASSERT_EQ(TTOpB.getTTranks(), TTOpC.getTTranks());
  for(int iDim = 0; iDim < TTOpB.tensorTrain().dimensions().size(); iDim++)
  {
    const auto& subTOpB = TTOpB.tensorTrain().subTensor(iDim);
    const auto& subTOpC = TTOpC.tensorTrain().subTensor(iDim);
    for(int i = 0; i < subTOpC.r1(); i++)
      for(int j = 0; j < 4; j++)
        for(int k = 0; k < 7; k++)
          for(int l = 0; l < subTOpC.r2(); l++)
            ASSERT_NEAR(subTOpB(i,TTOpB.index(iDim,j,k),l), subTOpC(i,TTOpC.index(iDim,j,k),l), eps);
  }
}


TEST(PITTS_TensorTrainOperator_apply_transposed_op, random)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOpA(4, 3, 5);
  randomize(TTOpA);

  TensorTrainOperator_double TTOpB(4, 3, 4);
  randomize(TTOpB);

  TensorTrainOperator_double TTOpC(4, 5, 4);
  applyT(TTOpA, TTOpB, TTOpC);

  TensorTrain_double TTx(4, 4, 3), TTy(4, 3), TTz(4, 5), TTz_ref(4, 5);
  randomize(TTx);

  apply(TTOpC, TTx, TTz);

  apply(TTOpB, TTx, TTy);
  applyT(TTOpA, TTy, TTz_ref);
  const auto error = axpby(-1., TTz_ref, 1., TTz);
  EXPECT_NEAR(0., error, eps);
}
