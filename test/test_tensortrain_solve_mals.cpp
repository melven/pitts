#include <gtest/gtest.h>
#include "pitts_tensortrain_solve_mals.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_random.hpp"

namespace
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  using mat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
  constexpr auto eps = 1.e-7;
}

TEST(PITTS_TensorTrain_solve_mals, Opeye_ones_nDim1)
{
  TensorTrainOperator_double TTOpA(1,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(1,5), TTb(1,5);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, true, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, Opeye_ones_nDim2)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, true, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, Opeye_ones_nDim6)
{
  TensorTrainOperator_double TTOpA(6,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(6,5), TTb(6,5);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, true, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, Opeye_ones_nDim6_nonsymmetric_least_squares)
{
  TensorTrainOperator_double TTOpA(6,5,4);
  TTOpA.setEye();
  TensorTrain_double TTx(6,4), TTb(6,5);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, false, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTx_ref(6,4);
  applyT(TTOpA, TTb, TTx_ref);

  double errNrm = axpby(-1., TTx_ref, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, random_nDim1)
{
  TensorTrainOperator_double TTOpA(1,5,5);
  randomize(TTOpA);
  TensorTrain_double TTx(1,5), TTb(1,5);
  randomize(TTb);
  randomize(TTx);

  double error = solveMALS(TTOpA, false, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, symmetric_random_nDim1)
{
  TensorTrainOperator_double TTOp_tmp(1,5,5);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(1,5,5);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  TensorTrain_double TTx(1,5), TTb(1,5);
  randomize(TTb);
  randomize(TTx);

  //std::cout << "Linear operator:\n";
  //for(int i = 0; i < 5; i++)
  //{
  //  for(int j = 0; j < 5; j++)
  //    std::cout << " " << TTOpA.tensorTrain().subTensors()[0](0,TTOpA.index(0,i,j),0);
  //  std::cout << "\n";
  //}
  double error = solveMALS(TTOpA, true, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, random_nDim2_rank1)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setTTranks(1);
  randomize(TTb);
  randomize(TTx);

  double error = solveMALS(TTOpA, false, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, random_nDim2)
{
  TensorTrainOperator_double TTOpA(2,2,2);
  TTOpA.setTTranks(2);
  randomize(TTOpA);
  TensorTrain_double TTx(2,2), TTb(2,2);
  TTb.setTTranks(2);
  randomize(TTb);
  TTx.setTTranks(2);
  randomize(TTx);

  double error = solveMALS(TTOpA, false, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, random_nDim6)
{
  TensorTrainOperator_double TTOpA(6,5,5);
  TTOpA.setTTranks(2);
  randomize(TTOpA);
  TensorTrain_double TTx(6,5), TTb(6,5);
  TTb.setTTranks(2);
  randomize(TTb);
  TTx.setTTranks(2);
  randomize(TTx);

  double error = solveMALS(TTOpA, false, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, random_nDim6_nonsymmetric_least_squares)
{
  TensorTrainOperator_double TTOpA(6,5,4);
  TTOpA.setTTranks(2);
  randomize(TTOpA);

  TensorTrain_double TTx(6,4), TTb(6,5);
  TTb.setTTranks(2);
  randomize(TTb);
  TTx.setTTranks(2);
  randomize(TTx);

  double error = solveMALS(TTOpA, false, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, symmetric_random_nDim6_rank1)
{
  TensorTrainOperator_double TTOp_tmp(6,5,4);
  TTOp_tmp.setTTranks(1);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(6,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);

  TensorTrain_double TTx(6,4), TTb(6,4), TTx_ref(6,4), TTr(6,4), TTdx(6,4);
  TTx_ref.setTTranks(1);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setOnes();

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveMALS(TTOpA, true, TTb, TTx, 5, eps, 10);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

TEST(PITTS_TensorTrain_solve_mals, symmetric_random_nDim6)
{
  TensorTrainOperator_double TTOp_tmp(6,5,4);
  TTOp_tmp.setTTranks(2);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(6,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);

  TensorTrain_double TTx(6,4), TTb(6,4), TTx_ref(6,4), TTr(6,4), TTdx(6,4);
  TTx_ref.setTTranks(2);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setTTranks(3);
  randomize(TTx);

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveMALS(TTOpA, true, TTb, TTx, 5, eps, 10);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

