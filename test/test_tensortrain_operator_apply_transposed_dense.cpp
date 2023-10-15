// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_operator_apply_transposed_dense.hpp"
#include "pitts_tensortrain_operator_apply_transposed.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"

namespace
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  constexpr auto eps = 1.e-10;
}

TEST(PITTS_TensorTrainOperator_apply_transposed_dense, zero)
{
  TensorTrainOperator_double TTOp(3, 5, 4);
  TTOp.setZero();

  MultiVector_double MVx(5*5*5,3), MVy;
  randomize(MVx);

  applyT(TTOp, MVx, MVy);

  ASSERT_EQ(4*4*4, MVy.rows());
  ASSERT_EQ(3, MVy.cols());
  for(int j = 0; j < 3; j++)
    for(int i = 0; i < 4*4*4; i++)
    {
      EXPECT_NEAR(0., MVy(i,j), eps);
    }
}


TEST(PITTS_TensorTrainOperator_apply_transposed_dense, eye)
{
  TensorTrainOperator_double TTOp(3, 5, 4);
  TTOp.setEye();

  MultiVector_double MVx(5*5*5,2), MVy;
  randomize(MVx);

  applyT(TTOp, MVx, MVy);

  ASSERT_EQ(4*4*4, MVy.rows());
  ASSERT_EQ(2, MVy.cols());
  for(int i = 0; i < 4; i++)
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 4; k++)
      {
        EXPECT_NEAR(MVx(i+j*5+k*25,0), MVy(i+j*4+k*16,0), eps);
        EXPECT_NEAR(MVx(i+j*5+k*25,1), MVy(i+j*4+k*16,1), eps);
      }
}

TEST(PITTS_TensorTrainOperator_apply_transposed_dense, random_nDim1)
{
  TensorTrainOperator_double TTOp(1, 5, 7);
  randomize(TTOp);
  const auto& subT = TTOp.tensorTrain().subTensor(0);

  MultiVector_double MVx(5,1), MVy;
  randomize(MVx);

  applyT(TTOp, MVx, MVy);

  ASSERT_EQ(7, MVy.rows());
  ASSERT_EQ(1, MVy.cols());
  for(int i = 0; i < 7; i++)
  {
    double yi_ref = 0;
    for(int j = 0; j < 5; j++)
      yi_ref += subT(0,TTOp.index(0,j,i),0) * MVx(j,0);
    EXPECT_NEAR(yi_ref, MVy(i,0), eps);
  }
}

TEST(PITTS_TensorTrainOperator_apply_transposed_dense, random_nDim2_rank1)
{
  TensorTrainOperator_double TTOp({2,5},{3,4});
  randomize(TTOp);

  TensorTrain_double TTx(std::vector<int>{2,5}), TTy(std::vector<int>{3,4});
  randomize(TTx);
  applyT(TTOp, TTx, TTy);

  MultiVector_double MVx(10,1), MVy, MVy_ref(12,1);
  toDense(TTx, &MVx(0,0), &MVx(10,0));
  toDense(TTy, &MVy_ref(0,0), &MVy_ref(12,0));

  applyT(TTOp, MVx, MVy);

  ASSERT_EQ(MVy_ref.rows(), MVy.rows());
  ASSERT_EQ(MVy_ref.cols(), MVy.cols());
  for(int i = 0; i < MVy_ref.rows(); i++)
    for(int j = 0; j < MVy_ref.cols(); j++)
    {
      EXPECT_NEAR(MVy_ref(i,j), MVy(i,j), eps);
    }
}

TEST(PITTS_TensorTrainOperator_apply_transposed_dense, random_nDim2)
{
  TensorTrainOperator_double TTOp({2,5},{3,4});
  TTOp.setTTranks(2);
  randomize(TTOp);

  TensorTrain_double TTx(std::vector<int>{2,5}), TTy(std::vector<int>{3,4});
  randomize(TTx);
  applyT(TTOp, TTx, TTy);

  MultiVector_double MVx(10,1), MVy, MVy_ref(12,1);
  toDense(TTx, &MVx(0,0), &MVx(10,0));
  toDense(TTy, &MVy_ref(0,0), &MVy_ref(12,0));

  applyT(TTOp, MVx, MVy);

  ASSERT_EQ(MVy_ref.rows(), MVy.rows());
  ASSERT_EQ(MVy_ref.cols(), MVy.cols());
  for(int i = 0; i < MVy_ref.rows(); i++)
    for(int j = 0; j < MVy_ref.cols(); j++)
    {
      EXPECT_NEAR(MVy_ref(i,j), MVy(i,j), eps);
    }
}

TEST(PITTS_TensorTrainOperator_apply_transposed_dense, random_nDim4_rank1)
{
  TensorTrainOperator_double TTOp({3,3,5,4},{3,2,3,4});
  randomize(TTOp);

  TensorTrain_double TTx(TTOp.row_dimensions()), TTy(TTOp.column_dimensions());
  randomize(TTx);
  applyT(TTOp, TTx, TTy);

  MultiVector_double MVx(3*3*5*4,1), MVy, MVy_ref(3*2*3*4,1);
  toDense(TTx, &MVx(0,0), &MVx(3*3*5*4,0));
  toDense(TTy, &MVy_ref(0,0), &MVy_ref(3*2*3*4,0));

  applyT(TTOp, MVx, MVy);

  ASSERT_EQ(MVy_ref.rows(), MVy.rows());
  ASSERT_EQ(MVy_ref.cols(), MVy.cols());
  for(int i = 0; i < MVy_ref.rows(); i++)
    for(int j = 0; j < MVy_ref.cols(); j++)
    {
      EXPECT_NEAR(MVy_ref(i,j), MVy(i,j), eps);
    }
}

TEST(PITTS_TensorTrainOperator_apply_transposed_dense, random_nDim4)
{
  TensorTrainOperator_double TTOp({3,3,5,4},{3,2,3,4});
  TTOp.setTTranks(3);
  randomize(TTOp);

  TensorTrain_double TTx(TTOp.row_dimensions()), TTy(TTOp.column_dimensions());
  randomize(TTx);
  applyT(TTOp, TTx, TTy);

  MultiVector_double MVx(3*3*5*4,1), MVy, MVy_ref(3*2*3*4,1);
  toDense(TTx, &MVx(0,0), &MVx(3*3*5*4,0));
  toDense(TTy, &MVy_ref(0,0), &MVy_ref(3*2*3*4,0));

  applyT(TTOp, MVx, MVy);

  ASSERT_EQ(MVy_ref.rows(), MVy.rows());
  ASSERT_EQ(MVy_ref.cols(), MVy.cols());
  for(int i = 0; i < MVy_ref.rows(); i++)
    for(int j = 0; j < MVy_ref.cols(); j++)
    {
      EXPECT_NEAR(MVy_ref(i,j), MVy(i,j), eps);
    }
}

