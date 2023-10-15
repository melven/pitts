// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_operator_to_dense.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_operator_apply_dense.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"

TEST(PITTS_TensorTrainOperator_toDense, scalar)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp(1, 1, 1);
  TTOp.setOnes();

  const Tensor2_double result = toDense(TTOp);
  ASSERT_EQ(1, result.r1());
  ASSERT_EQ(1, result.r2());
  EXPECT_NEAR(1., result(0,0), eps);
}

TEST(PITTS_TensorTrainOperator_toDense, unitMatrix1dTT)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp(1, 7, 5);
  TTOp.setEye();

  const Tensor2_double result = toDense(TTOp);
  Eigen::MatrixXd result_ref(7, 5);
  for(int i = 0; i < 7; i++)
    for(int j = 0; j < 5; j++)
      result_ref(i,j) = (i == j) ? 1. : 0.;
  EXPECT_NEAR(result_ref, ConstEigenMap(result), eps);
}

TEST(PITTS_TensorTrainOperator_toDense, unitMatrix2dTT)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp(2, 3, 3);
  TTOp.setEye();

  const Tensor2_double result = toDense(TTOp);
  const Eigen::MatrixXd result_ref = Eigen::MatrixXd::Identity(9, 9);
  EXPECT_NEAR(result_ref, ConstEigenMap(result), eps);
}

TEST(PITTS_TensorTrainOperator_toDense, randomRectangular2dTT)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp({3,7},{4,5});
  TTOp.setTTranks(2);
  randomize(TTOp);

  const Tensor2_double result = toDense(TTOp);
  std::vector<double> resultVec_ref(3*7*4*5);
  toDense(TTOp.tensorTrain(), resultVec_ref.begin(), resultVec_ref.end());

  Eigen::MatrixXd result_ref(3*7, 4*5);
  for(int i1 = 0; i1 < 3; i1++)
    for(int i2 = 0; i2 < 7; i2++)
      for(int j1 = 0; j1 < 4; j1++)
        for(int j2 = 0; j2 < 5; j2++)
          result_ref(i1+i2*3, j1+j2*4) = resultVec_ref[i1 + j1*3 + i2*3*4 + j2*3*4*7];
  
  EXPECT_NEAR(result_ref, ConstEigenMap(result), eps);
}

TEST(PITTS_TensorTrainOperator_toDense, randomRectangularMultiDim)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp({3,7,6,2},{4,5,2,3});
  TTOp.setTTranks({2,3,2});
  randomize(TTOp);

  const Tensor2_double denseOp = toDense(TTOp);

  // test with apply and random multivector
  MultiVector_double mvX(4*5*2*3,20);
  randomize(mvX);
  MultiVector_double mvResult(3*7*6*2,20);
  apply(TTOp, mvX, mvResult);

  const Eigen::MatrixXd result_ref = ConstEigenMap(denseOp) * ConstEigenMap(mvX);
  EXPECT_NEAR(result_ref, ConstEigenMap(mvResult), eps);
}