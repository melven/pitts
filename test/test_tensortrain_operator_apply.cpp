#include <gtest/gtest.h>
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_laplace_operator.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_tensortrain_operator_apply_dense.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"

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
    Tensor3_double subT = TTOpDummy.tensorTrain().setSubTensor(iDim, std::move(tridi));
    axpby(1., TTOpDummy, 1., TTOpLaplace);
    tridi = TTOpDummy.tensorTrain().setSubTensor(iDim, std::move(subT));
  }


  TensorTrain_double TTx(4, 5, 3), TTy(4, 5), TTy_ref(4, 5);
  randomize(TTx);

  apply(TTOpLaplace, TTx, TTy);
  copy(TTx, TTy_ref);
  const auto nrm_ref = laplaceOperator(TTy_ref);
  const auto error = axpby(-nrm_ref, TTy_ref, 1., TTy);
  EXPECT_NEAR(0., error, eps);
}

TEST(PITTS_TensorTrainOperator_apply, eye_boundaryRank)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp({3,4,2,5}, {3,4,2,5});
  TTOp.setEye();

  TensorTrain_double TTx(std::vector<int>{4,2}), TTy(std::vector<int>{4,2});
  TTx.setTTranks(2);
  Tensor3_double subT_l(3,4,2);
  Tensor3_double subT_r(2,2,5);
  TTx.setSubTensor(0, std::move(subT_l));
  TTx.setSubTensor(1, std::move(subT_r));
  randomize(TTx);

  apply(TTOp, TTx, TTy);

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

TEST(PITTS_TensorTrainOperator_apply, random_boundaryRank_rank1)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp({3,4,2,5}, {2,4,2,3});
  TTOp.setTTranks(1);
  randomize(TTOp);

  TensorTrain_double TTx(std::vector<int>{4,2}), TTy(std::vector<int>{4,2});
  Tensor3_double subT_l(2,4,1);
  Tensor3_double subT_r(1,2,3);
  TTx.setSubTensor(0, std::move(subT_l));
  TTx.setSubTensor(1, std::move(subT_r));
  randomize(TTx);

  apply(TTOp, TTx, TTy);

  ASSERT_EQ(3, TTy.subTensor(0).r1());
  ASSERT_EQ(5, TTy.subTensor(1).r2());

  // calculate a reference solution using TTOp * dense and toDense
  MultiVector_double MVx, MVy_ref, MVy;
  toDense(TTx, MVx);
  apply(TTOp, MVx, MVy_ref);
  toDense(TTy, MVy);

  EXPECT_NEAR(ConstEigenMap(MVy_ref), ConstEigenMap(MVy), eps);
}

TEST(PITTS_TensorTrainOperator_apply, random_boundaryRank)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp({3,4,2,5}, {2,4,2,3});
  TTOp.setTTranks(2);
  randomize(TTOp);

  TensorTrain_double TTx(std::vector<int>{4,2}), TTy(std::vector<int>{4,2});
  TTx.setTTranks(6);
  Tensor3_double subT_l(2,4,6);
  Tensor3_double subT_r(6,2,3);
  TTx.setSubTensor(0, std::move(subT_l));
  TTx.setSubTensor(1, std::move(subT_r));
  randomize(TTx);

  apply(TTOp, TTx, TTy);

  ASSERT_EQ(3, TTy.subTensor(0).r1());
  ASSERT_EQ(5, TTy.subTensor(1).r2());

  // calculate a reference solution using TTOp * dense and toDense
  MultiVector_double MVx, MVy_ref, MVy;
  toDense(TTx, MVx);
  apply(TTOp, MVx, MVy_ref);
  toDense(TTy, MVy);

  EXPECT_NEAR(ConstEigenMap(MVy_ref), ConstEigenMap(MVy), eps);
}
