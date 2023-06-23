#include <gtest/gtest.h>
#include "pitts_tensortrain_operator_to_qtt.hpp"
#include "pitts_tensortrain_operator_to_dense.hpp"
#include "pitts_tensortrain_operator_debug.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"

TEST(PITTS_TensorTrainOperator_toQtt, matrix_ones)
{
  using PITTS::debug::operator-;
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp(1, 8, 8);
  TTOp.setOnes();

  TensorTrainOperator_double qTTOp = toQtt(TTOp);
  TensorTrainOperator_double qTTOp_ref(3, 2, 2);
  qTTOp_ref.setOnes();

  EXPECT_NEAR(0., norm2((qTTOp - qTTOp_ref).tensorTrain()), eps);
}

TEST(PITTS_TensorTrainOperator_toQtt, multidim_random)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  constexpr auto eps = 1.e-10;

  TensorTrainOperator_double TTOp({4,6,4}, {4,6,4});
  TTOp.setTTranks(2);
  randomize(TTOp);

  TensorTrainOperator_double qTTOp = toQtt(TTOp);

  const Tensor2_double op = toDense(qTTOp);
  const Tensor2_double op_ref = toDense(TTOp);

  EXPECT_NEAR(ConstEigenMap(op_ref), ConstEigenMap(op), eps);
}