#include <gtest/gtest.h>
#include "pitts_tensortrain_solve_gmres.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_operator_apply_transposed_op.hpp"
#include "pitts_tensortrain_operator_apply_transposed.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "eigen_test_helper.hpp"

namespace
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-7;
}

TEST(PITTS_TensorTrain_solve_gmres, Opeye_ones_nDim1)
{
  TensorTrainOperator_double TTOpA(1,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(1,5), TTb(1,5);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveGMRES(TTOpA, TTb, TTx, 0, eps, eps, 999, true, false, "test: ", true);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_gmres, Opeye_ones_nDim2_guess_zero)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setOnes();
  TTx.setZero();

  double error = solveGMRES(TTOpA, TTb, TTx, 1, eps, eps, 999, true, false, "test: ", true);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_gmres, Opeye_ones_nDim2_guess_zero_symm)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setOnes();
  TTx.setZero();

  double error = solveGMRES(TTOpA, TTb, TTx, 1, eps, eps, 999, true, true, "test: ", true);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_gmres, random_nDim1)
{
  TensorTrainOperator_double TTOpA(1,5,5);
  randomize(TTOpA);
  TensorTrain_double TTx(1,5), TTb(1,5);
  randomize(TTb);
  randomize(TTx);

  double error = solveGMRES(TTOpA, TTb, TTx, 5, eps, eps, 999, true, false, "test: ", true);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_gmres, symmetric_random_nDim1_nonsymmAlg)
{
  TensorTrainOperator_double TTOp_tmp(1,5,5);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(1,5,5);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  TensorTrain_double TTx(1,5), TTb(1,5);
  randomize(TTb);
  randomize(TTx);

  double error = solveGMRES(TTOpA, TTb, TTx, 5, eps, eps, 999, true, false, "test: ", true);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_gmres, symmetric_random_nDim1_symmAlg)
{
  TensorTrainOperator_double TTOp_tmp(1,5,5);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(1,5,5);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  TensorTrain_double TTx(1,5), TTb(1,5);
  randomize(TTb);
  randomize(TTx);

  double error = solveGMRES(TTOpA, TTb, TTx, 5, eps, eps, 999, true, true, "test: ", true);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_gmres, random_nDim2_rank1)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  // make it diagonally dominant to obtain a well-posed problem
  for(int iDim = 0; iDim < 2; iDim++)
  {
    Tensor3_double subT;
    copy(TTOpA.tensorTrain().subTensor(iDim), subT);
    for(int i = 0; i < 5; i++)
      subT(0, TTOpA.index(iDim, i, i), 0) += 4;
    TTOpA.tensorTrain().setSubTensor(iDim, std::move(subT));
  }
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setTTranks(1);
  randomize(TTb);
  normalize(TTb);
  randomize(TTx);
  normalize(TTx);

  double error = solveGMRES(TTOpA, TTb, TTx, 25, eps, eps, 999, true, false, "test: ", true);
  EXPECT_NEAR(0, error, 100*eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, 10*eps);
}

TEST(PITTS_TensorTrain_solve_gmres, random_nDim2)
{
  TensorTrainOperator_double TTOpA(2,2,2);
  TTOpA.setTTranks(2);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(2,2,2);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/10, TTOpA);

  TensorTrain_double TTx(2,2), TTb(2,2);
  TTb.setTTranks(2);
  randomize(TTb);
  TTx.setOnes();

  double error = solveGMRES(TTOpA, TTb, TTx, 4, eps, eps, 999, true, false, "test: ", true);
  EXPECT_NEAR(0, error, 1.e-5*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, 1.e-5*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_gmres, symmetric_random_nDim6_rank1_nonsymmAlg)
{
  TensorTrainOperator_double TTOp_tmp(6,5,4);
  TTOp_tmp.setTTranks(1);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(6,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  // make it diagonally dominant to obtain a well-posed problem
  for(int iDim = 0; iDim < 6; iDim++)
  {
    Tensor3_double subT;
    copy(TTOpA.tensorTrain().subTensor(iDim), subT);
    for(int i = 0; i < 4; i++)
      subT(0, TTOpA.index(iDim, i, i), 0) += 50;
    TTOpA.tensorTrain().setSubTensor(iDim, std::move(subT));
  }

  TensorTrain_double TTx(6,4), TTb(6,4), TTx_ref(6,4), TTr(6,4), TTdx(6,4);
  TTx_ref.setTTranks(1);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setOnes();

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveGMRES(TTOpA, TTb, TTx, 25, 0.01, 0.01, 999, true, false, "test: ", true);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, 0.01*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

TEST(PITTS_TensorTrain_solve_gmres, symmetric_random_nDim6_rank1_symmAlg)
{
  TensorTrainOperator_double TTOp_tmp(6,5,4);
  TTOp_tmp.setTTranks(1);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(6,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  // make it diagonally dominant to obtain a well-posed problem
  for(int iDim = 0; iDim < 6; iDim++)
  {
    Tensor3_double subT;
    copy(TTOpA.tensorTrain().subTensor(iDim), subT);
    for(int i = 0; i < 4; i++)
      subT(0, TTOpA.index(iDim, i, i), 0) += 50;
    TTOpA.tensorTrain().setSubTensor(iDim, std::move(subT));
  }

  TensorTrain_double TTx(6,4), TTb(6,4), TTx_ref(6,4), TTr(6,4), TTdx(6,4);
  TTx_ref.setTTranks(1);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setOnes();

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveGMRES(TTOpA, TTb, TTx, 25, 0.01, 0.01, 999, true, true, "test: ", true);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, 0.01*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

TEST(PITTS_TensorTrain_solve_gmres, symmetric_random_nDim6_nonsymmAlg)
{
  TensorTrainOperator_double TTOp_tmp(6,5,4);
  TTOp_tmp.setTTranks(2);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(6,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  // make it diagonally dominant to obtain a well-posed problem
  for(int iDim = 0; iDim < 6; iDim++)
  {
    Tensor3_double subT;
    copy(TTOpA.tensorTrain().subTensor(iDim), subT);
    for(int i = 0; i < 4; i++)
      subT(0, TTOpA.index(iDim, i, i), 0) += 50;
    TTOpA.tensorTrain().setSubTensor(iDim, std::move(subT));
  }

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


  double residualNorm = solveGMRES(TTOpA, TTb, TTx, 25, 0.01, 0.01, 999, true, false, "test: ", true);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, 0.01*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

TEST(PITTS_TensorTrain_solve_gmres, symmetric_random_nDim6_symmAlg)
{
  TensorTrainOperator_double TTOp_tmp(6,5,4);
  TTOp_tmp.setTTranks(2);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(6,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  // make it diagonally dominant to obtain a well-posed problem
  for(int iDim = 0; iDim < 6; iDim++)
  {
    Tensor3_double subT;
    copy(TTOpA.tensorTrain().subTensor(iDim), subT);
    for(int i = 0; i < 4; i++)
      subT(0, TTOpA.index(iDim, i, i), 0) += 50;
    TTOpA.tensorTrain().setSubTensor(iDim, std::move(subT));
  }

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


  double residualNorm = solveGMRES(TTOpA, TTb, TTx, 25, 0.01, 0.01, 999, true, true, "test: ", true);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, 0.01*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}
