#include <gtest/gtest.h>
#include "pitts_tensortrain_operator_apply_dense.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_norm.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"

namespace
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  using TTOpDenseHelper_double = PITTS::TTOpApplyDenseHelper<double>;
  constexpr auto eps = 1.e-10;
}

TEST(PITTS_TensorTrainOperator_apply_dense, zero)
{
  TensorTrainOperator_double TTOp(3, 4, 5);
  TTOp.setZero();

  MultiVector_double MVx(5*5*5,3), MVy;
  randomize(MVx);

  apply(TTOp, MVx, MVy);

  ASSERT_EQ(4*4*4, MVy.rows());
  ASSERT_EQ(3, MVy.cols());
  for(int j = 0; j < 3; j++)
    for(int i = 0; i < 4*4*4; i++)
    {
      EXPECT_NEAR(0., MVy(i,j), eps);
    }
}


TEST(PITTS_TensorTrainOperator_apply_dense, eye)
{
  TensorTrainOperator_double TTOp(3, 4, 5);
  TTOp.setEye();

  MultiVector_double MVx(5*5*5,2), MVy;
  randomize(MVx);

  apply(TTOp, MVx, MVy);

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

TEST(PITTS_TensorTrainOperator_apply_dense, random_nDim1)
{
  TensorTrainOperator_double TTOp(1, 7, 5);
  randomize(TTOp);
  const auto& subT = TTOp.tensorTrain().subTensor(0);

  MultiVector_double MVx(5,1), MVy;
  randomize(MVx);

  apply(TTOp, MVx, MVy);

  ASSERT_EQ(7, MVy.rows());
  ASSERT_EQ(1, MVy.cols());
  for(int i = 0; i < 7; i++)
  {
    double yi_ref = 0;
    for(int j = 0; j < 5; j++)
      yi_ref += subT(0,TTOp.index(0,i,j),0) * MVx(j,0);
    EXPECT_NEAR(yi_ref, MVy(i,0), eps);
  }
}

TEST(PITTS_TensorTrainOperator_apply_dense, random_nDim2_rank1)
{
  TensorTrainOperator_double TTOp({3,4},{2,5});
  randomize(TTOp);

  TensorTrain_double TTx(std::vector<int>{2,5}), TTy(std::vector<int>{3,4});
  randomize(TTx);
  apply(TTOp, TTx, TTy);

  MultiVector_double MVx(10,1), MVy, MVy_ref(12,1);
  toDense(TTx, &MVx(0,0), &MVx(10,0));
  toDense(TTy, &MVy_ref(0,0), &MVy_ref(12,0));

  apply(TTOp, MVx, MVy);

  ASSERT_EQ(MVy_ref.rows(), MVy.rows());
  ASSERT_EQ(MVy_ref.cols(), MVy.cols());
  for(int i = 0; i < MVy_ref.rows(); i++)
    for(int j = 0; j < MVy_ref.cols(); j++)
    {
      EXPECT_NEAR(MVy_ref(i,j), MVy(i,j), eps);
    }
}

TEST(PITTS_TensorTrainOperator_apply_dense, random_nDim2)
{
  TensorTrainOperator_double TTOp({3,4},{2,5});
  TTOp.setTTranks(2);
  randomize(TTOp);

  TensorTrain_double TTx(std::vector<int>{2,5}), TTy(std::vector<int>{3,4});
  randomize(TTx);
  apply(TTOp, TTx, TTy);

  MultiVector_double MVx(10,1), MVy, MVy_ref(12,1);
  toDense(TTx, &MVx(0,0), &MVx(10,0));
  toDense(TTy, &MVy_ref(0,0), &MVy_ref(12,0));

  apply(TTOp, MVx, MVy);

  ASSERT_EQ(MVy_ref.rows(), MVy.rows());
  ASSERT_EQ(MVy_ref.cols(), MVy.cols());
  for(int i = 0; i < MVy_ref.rows(); i++)
    for(int j = 0; j < MVy_ref.cols(); j++)
    {
      EXPECT_NEAR(MVy_ref(i,j), MVy(i,j), eps);
    }
}

TEST(PITTS_TensorTrainOperator_apply_dense, random_nDim4_rank1)
{
  TensorTrainOperator_double TTOp({3,2,3,4},{3,3,5,4});
  randomize(TTOp);

  TensorTrain_double TTx(TTOp.column_dimensions()), TTy(TTOp.row_dimensions());
  randomize(TTx);
  apply(TTOp, TTx, TTy);

  MultiVector_double MVx(3*3*5*4,1), MVy, MVy_ref(3*2*3*4,1);
  toDense(TTx, &MVx(0,0), &MVx(3*3*5*4,0));
  toDense(TTy, &MVy_ref(0,0), &MVy_ref(3*2*3*4,0));

  apply(TTOp, MVx, MVy);

  ASSERT_EQ(MVy_ref.rows(), MVy.rows());
  ASSERT_EQ(MVy_ref.cols(), MVy.cols());
  for(int i = 0; i < MVy_ref.rows(); i++)
    for(int j = 0; j < MVy_ref.cols(); j++)
    {
      EXPECT_NEAR(MVy_ref(i,j), MVy(i,j), eps);
    }
}

TEST(PITTS_TensorTrainOperator_apply_dense, random_nDim4)
{
  TensorTrainOperator_double TTOp({3,2,3,4},{3,3,5,4});
  TTOp.setTTranks(3);
  randomize(TTOp);

  TensorTrain_double TTx(TTOp.column_dimensions()), TTy(TTOp.row_dimensions());
  randomize(TTx);
  apply(TTOp, TTx, TTy);

  MultiVector_double MVx(3*3*5*4,1), MVy, MVy_ref(3*2*3*4,1);
  toDense(TTx, &MVx(0,0), &MVx(3*3*5*4,0));
  toDense(TTy, &MVy_ref(0,0), &MVy_ref(3*2*3*4,0));

  apply(TTOp, MVx, MVy);

  ASSERT_EQ(MVy_ref.rows(), MVy.rows());
  ASSERT_EQ(MVy_ref.cols(), MVy.cols());
  for(int i = 0; i < MVy_ref.rows(); i++)
    for(int j = 0; j < MVy_ref.cols(); j++)
    {
      EXPECT_NEAR(MVy_ref(i,j), MVy(i,j), eps);
    }
}

TEST(PITTS_TensorTrainOperator_apply_dense, TTOpHelper_apply_addRemovePadding)
{
  const std::vector dims = {5, 7, 2, 3};
  const auto N = 5*7*2*3;
  const auto nlast = dims[3];
  TensorTrainOperator_double TTOp(dims, dims);
  TTOp.setTTranks({2,1,3});
  randomize(TTOp);

  MultiVector_double MVx(N, 1), MVx_ref;
  randomize(MVx);
  copy(MVx, MVx_ref);

  TTOpDenseHelper_double TTOpHelper(TTOp);

  TTOpHelper.addPadding(MVx);
  EXPECT_EQ(1, MVx.cols());
  EXPECT_GT(MVx.rows(), MVx_ref.rows());
  // padding should be zero!
  const double nrm_ref = norm2(MVx_ref)(0);
  const double nrm = norm2(MVx)(0);
  EXPECT_NEAR(nrm_ref, nrm, eps);

  // entries should be the same when reshaped/reordered appropriately (todo)

  TTOpHelper.removePadding(MVx);
  EXPECT_NEAR(ConstEigenMap(MVx_ref), ConstEigenMap(MVx), eps);
}

TEST(PITTS_TensorTrainOperator_apply_dense, TTOpHelper_apply)
{
  const std::vector dims = {3, 4, 2, 3};
  const auto N = 3*4*2*3;

  TensorTrainOperator_double TTOp(dims, dims);
  TTOp.setTTranks({2,1,3});
  randomize(TTOp);

  MultiVector_double MVx(N, 1), MVx_ref, MVy, MVy_ref;
  randomize(MVx);
  copy(MVx, MVx_ref);

  TTOpDenseHelper_double TTOpHelper(TTOp);
  TTOpHelper.addPadding(MVx);

  apply(TTOpHelper, MVx, MVy);

  TTOpHelper.removePadding(MVy);

  apply(TTOp, MVx_ref, MVy_ref);
  EXPECT_NEAR(ConstEigenMap(MVy_ref), ConstEigenMap(MVy), eps);
}

TEST(PITTS_TensorTrainOperator_apply_dense, TTOpHelper_apply_large)
{
  const std::vector dims = {17, 5, 39};
  const auto N = 17*5*39;

  TensorTrainOperator_double TTOp(dims, dims);
  TTOp.setTTranks({4,3});
  randomize(TTOp);

  MultiVector_double MVx(N, 1), MVx_ref, MVy, MVy_ref;
  randomize(MVx);
  copy(MVx, MVx_ref);

  TTOpDenseHelper_double TTOpHelper(TTOp);
  TTOpHelper.addPadding(MVx);

  apply(TTOpHelper, MVx, MVy);

  TTOpHelper.removePadding(MVy);

  apply(TTOp, MVx_ref, MVy_ref);
  EXPECT_NEAR(ConstEigenMap(MVy_ref), ConstEigenMap(MVy), eps);
}