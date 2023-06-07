#include <gtest/gtest.h>
#include "pitts_tensortrain_solve_mals.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_operator_apply_transposed.hpp"
#include "pitts_tensortrain_operator_apply_op.hpp"
#include "pitts_tensortrain_operator_apply_transposed_op.hpp"
#include "pitts_tensortrain_random.hpp"
#include "eigen_test_helper.hpp"

namespace
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  using mat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
  using PITTS::MALS_projection;
  constexpr auto eps = 1.e-7;
}

TEST(PITTS_TensorTrain_solve_mals, MALS_Opeye_ones_nDim1)
{
  TensorTrainOperator_double TTOpA(1,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(1,5), TTb(1,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_Opeye_ones_nDim2)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 1, eps, 10, 1, 0);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, simplified_AMEn_Opeye_ones_nDim2)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 1, eps, 10, 1, 0, 1, true);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_Opeye_ones_nDim2)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 1, eps, 10, 1, 0, 1, false);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_Opeye_ones_nDim6)
{
  TensorTrainOperator_double TTOpA(6,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(6,5), TTb(6,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 1, eps, 10, 1, 0);
  EXPECT_NEAR(0, error, eps*norm2(TTb));

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, simplified_AMEn_Opeye_ones_nDim6)
{
  TensorTrainOperator_double TTOpA(6,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(6,5), TTb(6,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 1, eps, 10, 1, 0, 1, true);
  EXPECT_NEAR(0, error, eps*norm2(TTb));

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_Opeye_ones_nDim6)
{
  TensorTrainOperator_double TTOpA(6,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(6,5), TTb(6,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 1, eps, 10, 1, 0, 1, false);
  EXPECT_NEAR(0, error, eps*norm2(TTb));

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, MALS_Opeye_ones_nDim6)
{
  TensorTrainOperator_double TTOpA(6,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(6,5), TTb(6,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps*norm2(TTb));

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, MALS_Opeye_ones_nDim1_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(1,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(1,5), TTb(1,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_Opeye_ones_nDim2_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10, 1, 0);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, simplified_AMEn_Opeye_ones_nDim2_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10, 1, 0, 1, true);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_Opeye_ones_nDim2_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10, 1, 0, 1, false);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, MALS_Opeye_ones_nDim2_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_Opeye_ones_nDim6_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(6,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(6,5), TTb(6,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10, 1, 0);
  EXPECT_NEAR(0, error, eps*norm2(TTb));

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, simplified_AMEn_Opeye_ones_nDim6_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(6,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(6,5), TTb(6,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10, 1, 0, 1, true);
  EXPECT_NEAR(0, error, eps*norm2(TTb));

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_Opeye_ones_nDim6_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(6,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(6,5), TTb(6,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10, 1, 0, 1, false);
  EXPECT_NEAR(0, error, eps*norm2(TTb));

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, MALS_Opeye_ones_nDim6_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(6,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(6,5), TTb(6,5);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps*norm2(TTb));

  double errNrm = axpby(-1., TTb, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, ALS_Opeye_ones_nDim6_nonsymmetric_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(6,4,4);
  TTOpA.setEye();
  TensorTrain_double TTx(6,4), TTb(6,4);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10, 1, 0);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTx_ref(6,4);
  applyT(TTOpA, TTb, TTx_ref);

  double errNrm = axpby(-1., TTx_ref, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, simplified_AMEn_Opeye_ones_nDim6_nonsymmetric_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(6,4,4);
  TTOpA.setEye();
  TensorTrain_double TTx(6,4), TTb(6,4);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10, 1, 0, 1, true);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTx_ref(6,4);
  applyT(TTOpA, TTb, TTx_ref);

  double errNrm = axpby(-1., TTx_ref, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_Opeye_ones_nDim6_nonsymmetric_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(6,4,4);
  TTOpA.setEye();
  TensorTrain_double TTx(6,4), TTb(6,4);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10, 1, 0, 1, false);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTx_ref(6,4);
  applyT(TTOpA, TTb, TTx_ref);

  double errNrm = axpby(-1., TTx_ref, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, MALS_Opeye_ones_nDim6_nonsymmetric_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(6,4,4);
  TTOpA.setEye();
  TensorTrain_double TTx(6,4), TTb(6,4);
  TTb.setOnes();
  randomize(TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::PetrovGalerkin, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTx_ref(6,4);
  applyT(TTOpA, TTb, TTx_ref);

  double errNrm = axpby(-1., TTx_ref, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_Opeye_ones_nDim6_nonsymmetric_least_squares)
{
  TensorTrainOperator_double TTOpA(6,5,4);
  TTOpA.setEye();
  TensorTrain_double TTx(6,4), TTb(6,5);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 1, eps, 10, 1, 0);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTx_ref(6,4);
  applyT(TTOpA, TTb, TTx_ref);

  double errNrm = axpby(-1., TTx_ref, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_Opeye_ones_nDim6_nonsymmetric_least_squares)
{
  TensorTrainOperator_double TTOpA(6,5,4);
  TTOpA.setEye();
  TensorTrain_double TTx(6,4), TTb(6,5);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 1, eps, 10, 1, 0, 1);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTx_ref(6,4);
  applyT(TTOpA, TTb, TTx_ref);

  double errNrm = axpby(-1., TTx_ref, 1., TTx);
  EXPECT_NEAR(0, errNrm, eps);
}

TEST(PITTS_TensorTrain_solve_mals, MALS_Opeye_ones_nDim6_nonsymmetric_least_squares)
{
  TensorTrainOperator_double TTOpA(6,5,4);
  TTOpA.setEye();
  TensorTrain_double TTx(6,4), TTb(6,5);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 1, eps, 10);
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

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 1, eps, 10);
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

  double error = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_random_nDim2_rank1)
{
  TensorTrainOperator_double TTOpA(2,5,5);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  TensorTrain_double TTx(2,5), TTb(2,5);
  TTb.setTTranks(1);
  randomize(TTb);
  normalize(TTb);
  randomize(TTx);
  normalize(TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 1, eps, 2, 1, 0);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  TensorTrain_double TTAtAx(TTb.dimensions());
  applyT(TTOpA, TTAx, TTAtAx);
  TensorTrain_double TTAtb(TTb.dimensions());
  applyT(TTOpA, TTb, TTAtb);
  double error_ref = axpby(-1., TTAtb, 1., TTAtAx);
  EXPECT_NEAR(error_ref, error, 0.01*eps);
}

TEST(PITTS_TensorTrain_solve_mals, simplified_AMEn_random_nDim2_rank1)
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

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 5, eps, 2, 1, 0, 1, true);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  TensorTrain_double TTAtAx(TTb.dimensions());
  applyT(TTOpA, TTAx, TTAtAx);
  TensorTrain_double TTAtb(TTb.dimensions());
  applyT(TTOpA, TTb, TTAtb);
  double error_ref = axpby(-1., TTAtb, 1., TTAtAx);
  //EXPECT_NEAR(error_ref, error, 0.01*eps);
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_random_nDim2_rank1)
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

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 5, eps, 2, 1, 0, 1, false);
  EXPECT_NEAR(0, error, eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  TensorTrain_double TTAtAx(TTb.dimensions());
  applyT(TTOpA, TTAx, TTAtAx);
  TensorTrain_double TTAtb(TTb.dimensions());
  applyT(TTOpA, TTb, TTAtb);
  double error_ref = axpby(-1., TTAtb, 1., TTAtAx);
  EXPECT_NEAR(error_ref, error, 0.01*eps);
}

TEST(PITTS_TensorTrain_solve_mals, MALS_random_nDim2_rank1)
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

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 5, eps, 3);
  EXPECT_NEAR(0, error, 100*eps);

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  TensorTrain_double TTAtAx(TTb.dimensions());
  applyT(TTOpA, TTAx, TTAtAx);
  TensorTrain_double TTAtb(TTb.dimensions());
  applyT(TTOpA, TTb, TTAtb);
  double error_ref = axpby(-1., TTAtb, 1., TTAtAx);
  EXPECT_NEAR(error_ref, error, 0.01*eps);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_random_nDim2)
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
  copy(TTb, TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 2, eps, 10, 1, 0);
  EXPECT_NEAR(0, error, 1.e-5*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, 1.e-4*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, simplified_AMEn_random_nDim2)
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
  copy(TTb, TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 2, eps, 10, 1, 0, 1, true);
  EXPECT_NEAR(0, error, 1.e-5*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  //EXPECT_NEAR(error_ref, error, 1.e-4*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_random_nDim2)
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
  copy(TTb, TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 2, eps, 10, 1, 0, 1, false);
  EXPECT_NEAR(0, error, 1.e-5*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, 1.e-4*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, MALS_random_nDim2)
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

  double error = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 1, eps, 10);
  EXPECT_NEAR(0, error, 1.e-5*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, 1.e-5*norm2(TTb));
}

TEST(PITTS_TensorTrain_solve_mals, ALS_random_nDim6_nonsymmetric_RitzGalerkin)
{
  TensorTrainOperator_double TTOpA(6,4,4);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(6,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/100, TTOpA);
  

  TensorTrain_double TTx(6,4), TTb(6,4);
  TTb.setOnes();
  TTx.setTTranks(2);
  randomize(TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::RitzGalerkin, TTb, TTx, 3, eps, 5, 1, 0);
  EXPECT_NEAR(0, error, 0.05*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, simplified_AMEn_random_nDim6_nonsymmetric_RitzGalerkin)
{
  TensorTrainOperator_double TTOpA(6,4,4);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(6,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/100, TTOpA);
  

  TensorTrain_double TTx(6,4), TTb(6,4);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, false, MALS_projection::RitzGalerkin, TTb, TTx, 3, eps, 5, 1, 0, 1, true);
  EXPECT_NEAR(0, error, 0.05*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  //EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_random_nDim6_nonsymmetric_RitzGalerkin)
{
  TensorTrainOperator_double TTOpA(6,4,4);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(6,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/100, TTOpA);
  

  TensorTrain_double TTx(6,4), TTb(6,4);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, false, MALS_projection::RitzGalerkin, TTb, TTx, 3, eps, 5, 1, 0, 1, false);
  EXPECT_NEAR(0, error, 0.05*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_random_nDim6_nonsymmetric_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(6,4,4);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(6,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/100, TTOpA);
  

  TensorTrain_double TTx(6,4), TTb(6,4);
  TTb.setOnes();
  TTx.setTTranks(2);
  randomize(TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::PetrovGalerkin, TTb, TTx, 3, eps, 5, 1, 0);
  EXPECT_NEAR(0, error, 0.05*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_random_nDim3_nonsymmetric_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(3,2,2);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(3,2,2);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/100, TTOpA);
  

  TensorTrain_double TTx(3,2), TTb(3,2);
  TTb.setOnes();
  TTx.setTTranks(2);
  randomize(TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::PetrovGalerkin, TTb, TTx, 2, eps, 5, 1, 0);
  EXPECT_NEAR(0, error, 0.05*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, simplified_AMEn_random_nDim3_nonsymmetric_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(3,2,2);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(3,2,2);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/100, TTOpA);
  

  TensorTrain_double TTx(3,2), TTb(3,2);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, false, MALS_projection::PetrovGalerkin, TTb, TTx, 2, eps, 5, 1, 0, 2, true);
  EXPECT_NEAR(0, error, 0.05*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  //EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_random_nDim3_nonsymmetric_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(3,2,2);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(3,2,2);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/100, TTOpA);
  

  TensorTrain_double TTx(3,2), TTb(3,2);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, false, MALS_projection::PetrovGalerkin, TTb, TTx, 2, eps, 5, 1, 0, 2, false);
  EXPECT_NEAR(0, error, 0.05*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}


TEST(PITTS_TensorTrain_solve_mals, MALS_random_nDim6_nonsymmetric_RitzGalerkin)
{
  TensorTrainOperator_double TTOpA(6,4,4);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(6,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/100, TTOpA);
  

  TensorTrain_double TTx(6,4), TTb(6,4);
  TTb.setOnes();
  TTx.setTTranks(3);
  randomize(TTx);

  double error = solveMALS(TTOpA, false, MALS_projection::RitzGalerkin, TTb, TTx, 3, eps, 5);
  EXPECT_NEAR(0, error, 0.005*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, MALS_random_nDim6_nonsymmetric_PetrovGalerkin)
{
  TensorTrainOperator_double TTOpA(6,4,4);
  TTOpA.setTTranks(1);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(6,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/100, TTOpA);
  

  TensorTrain_double TTx(6,4), TTb(6,4);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, false, MALS_projection::PetrovGalerkin, TTb, TTx, 2, eps, 5);
  EXPECT_NEAR(0, error, 0.005*norm2(TTb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  double error_ref = axpby(-1., TTb, 1., TTAx);
  EXPECT_NEAR(error_ref, error, eps);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_random_nDim6_nonsymmetric_least_squares)
{
  TensorTrainOperator_double TTOpA(6,5,4);
  TTOpA.setTTranks(2);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(6,5,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/100, TTOpA);
  

  TensorTrain_double TTx(6,4), TTb(6,5);
  TTb.setOnes();
  TTx.setTTranks(3);
  randomize(TTx);

  double normalResidual = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 3, eps, 5, 1, 0);
  TensorTrain_double TTAtb(TTx.dimensions());
  applyT(TTOpA, TTb, TTAtb);
  EXPECT_NEAR(0, normalResidual, 0.5*norm2(TTAtb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  TensorTrain_double TTAtAx(TTx.dimensions());
  applyT(TTOpA, TTAx, TTAtAx);
  const double normalResidual_ref = axpby(-1., TTAtb, 1., TTAtAx);
  EXPECT_NEAR(normalResidual_ref, normalResidual, eps*norm2(TTAtb));

  //const double residualError = axpby(-1., TTb, 1., TTAx);
  //EXPECT_NEAR(0., residualError, 0.1);
}

TEST(PITTS_TensorTrain_solve_mals, MALS_random_nDim6_nonsymmetric_least_squares)
{
  TensorTrainOperator_double TTOpA(6,5,4);
  TTOpA.setTTranks(2);
  randomize(TTOpA);
  normalize(TTOpA);
  TensorTrainOperator_double TTOpI(6,5,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/10, TTOpA);
  

  TensorTrain_double TTx(6,4), TTb(6,5);
  TTb.setOnes();
  TTx.setOnes();

  double normalResidual = solveMALS(TTOpA, false, MALS_projection::NormalEquations, TTb, TTx, 1, eps, 10);
  TensorTrain_double TTAtb(TTx.dimensions());
  applyT(TTOpA, TTb, TTAtb);
  EXPECT_NEAR(0, normalResidual, 0.5*norm2(TTAtb));

  TensorTrain_double TTAx(TTb.dimensions());
  apply(TTOpA, TTx, TTAx);
  TensorTrain_double TTAtAx(TTx.dimensions());
  applyT(TTOpA, TTAx, TTAtAx);
  const double normalResidual_ref = axpby(-1., TTAtb, 1., TTAtAx);
  EXPECT_NEAR(normalResidual_ref, normalResidual, eps*norm2(TTAtb));

  //const double residualError = axpby(-1., TTb, 1., TTAx);
  //EXPECT_NEAR(0., residualError, 0.1);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_symmetric_random_nDim6_rank1)
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
      subT(0, TTOpA.index(iDim, i, i), 0) += 4;
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


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 5, eps, 2, 1, 0);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

TEST(PITTS_TensorTrain_solve_mals, simplified_AMEn_symmetric_random_nDim6_rank1)
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
      subT(0, TTOpA.index(iDim, i, i), 0) += 4;
    TTOpA.tensorTrain().setSubTensor(iDim, std::move(subT));
  }
  normalize(TTOpA);

  TensorTrainOperator_double TTOpI(6,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/20, TTOpA);

  TensorTrain_double TTx(6,4), TTb(6,4), TTx_ref(6,4), TTr(6,4), TTdx(6,4);
  TTx_ref.setTTranks(1);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setOnes();

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 5, eps, 2, 1, 0, 1, true);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_symmetric_random_nDim6_rank1)
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
      subT(0, TTOpA.index(iDim, i, i), 0) += 4;
    TTOpA.tensorTrain().setSubTensor(iDim, std::move(subT));
  }
  normalize(TTOpA);

  TensorTrain_double TTx(6,4), TTb(6,4), TTx_ref(6,4), TTr(6,4), TTdx(6,4);
  TTx_ref.setTTranks(1);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setOnes();

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 5, eps, 2, 1, 0, 1, false);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

TEST(PITTS_TensorTrain_solve_mals, MALS_symmetric_random_nDim6_rank1)
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
      subT(0, TTOpA.index(iDim, i, i), 0) += 2;
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


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 5, eps, 2);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.02);
}

TEST(PITTS_TensorTrain_solve_mals, MALS_symmetric_random_nDim6_rank1_PetrovGalerkin)
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
      subT(0, TTOpA.index(iDim, i, i), 0) += 2;
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


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 2, eps, 2);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.02);
}

TEST(PITTS_TensorTrain_solve_mals, ALS_symmetric_random_nDim5)
{
  TensorTrainOperator_double TTOp_tmp(5,5,4);
  TTOp_tmp.setTTranks(2);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(5,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  normalize(TTOpA);

  TensorTrainOperator_double TTOpI(5,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/20, TTOpA);

  TensorTrain_double TTx(5,4), TTb(5,4), TTx_ref(5,4), TTr(5,4), TTdx(5,4);
  TTx_ref.setTTranks(2);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setTTranks(3);
  randomize(TTx);

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 5, eps, 10, 1, 0);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

TEST(PITTS_TensorTrain_solve_mals, simplfied_AMEn_symmetric_random_nDim5)
{
  TensorTrainOperator_double TTOp_tmp(5,5,4);
  TTOp_tmp.setTTranks(2);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(5,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  normalize(TTOpA);

  TensorTrainOperator_double TTOpI(5,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/20, TTOpA);

  TensorTrain_double TTx(5,4), TTb(5,4), TTx_ref(5,4), TTr(5,4), TTdx(5,4);
  TTx_ref.setTTranks(2);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setOnes();

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 5, eps, 10, 1, 0, 1, true);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

TEST(PITTS_TensorTrain_solve_mals, AMEn_symmetric_random_nDim5)
{
  TensorTrainOperator_double TTOp_tmp(5,5,4);
  TTOp_tmp.setTTranks(2);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(5,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  normalize(TTOpA);

  TensorTrainOperator_double TTOpI(5,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/20, TTOpA);

  TensorTrain_double TTx(5,4), TTb(5,4), TTx_ref(5,4), TTr(5,4), TTdx(5,4);
  TTx_ref.setTTranks(2);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setOnes();

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 5, eps, 10, 1, 0, 1, false);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

TEST(PITTS_TensorTrain_solve_mals, DISABLED_ALS_symmetric_random_nDim5_PetrovGalerkin)
{
  TensorTrainOperator_double TTOp_tmp(5,5,4);
  TTOp_tmp.setTTranks(2);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(5,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  normalize(TTOpA);

  TensorTrainOperator_double TTOpI(5,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/20, TTOpA);

  TensorTrain_double TTx(5,4), TTb(5,4), TTx_ref(5,4), TTr(5,4), TTdx(5,4);
  TTx_ref.setTTranks(2);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setTTranks(1);
  randomize(TTx);

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 5, eps, 10, 1, 0);
  EXPECT_NEAR(0, residualNorm, 0.05*norm2(TTb));


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);
}

TEST(PITTS_TensorTrain_solve_mals, DISABLED_AMEn_symmetric_random_nDim5_PetrovGalerkin)
{
  TensorTrainOperator_double TTOp_tmp(5,5,4);
  TTOp_tmp.setTTranks(2);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(5,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  normalize(TTOpA);

  TensorTrainOperator_double TTOpI(5,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/20, TTOpA);

  TensorTrain_double TTx(5,4), TTb(5,4), TTx_ref(5,4), TTr(5,4), TTdx(5,4);
  TTx_ref.setTTranks(2);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setOnes();

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::PetrovGalerkin, TTb, TTx, 5, eps, 10, 1, 0, 1);
  EXPECT_NEAR(0, residualNorm, 0.05*norm2(TTb));


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);
}

TEST(PITTS_TensorTrain_solve_mals, MALS_symmetric_random_nDim5)
{
  TensorTrainOperator_double TTOp_tmp(5,5,4);
  TTOp_tmp.setTTranks(2);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(5,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  normalize(TTOpA);

  TensorTrainOperator_double TTOpI(5,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/20, TTOpA);

  TensorTrain_double TTx(5,4), TTb(5,4), TTx_ref(5,4), TTr(5,4), TTdx(5,4);
  TTx_ref.setTTranks(2);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setOnes();

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 1, eps, 10, 2, 1, 0, false, false, 25, eps);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}

TEST(PITTS_TensorTrain_solve_mals, DISABLED_MALS_symmetric_random_nDim5_with_TTgmres)
{
  TensorTrainOperator_double TTOp_tmp(5,5,4);
  TTOp_tmp.setTTranks(2);
  randomize(TTOp_tmp);
  TensorTrainOperator_double TTOpA(5,4,4);
  applyT(TTOp_tmp, TTOp_tmp, TTOpA);
  normalize(TTOpA);

  TensorTrainOperator_double TTOpI(5,4,4);
  TTOpI.setEye();
  const double Inrm = normalize(TTOpI);
  axpby(Inrm, TTOpI, Inrm/20, TTOpA);

  TensorTrain_double TTx(5,4), TTb(5,4), TTx_ref(5,4), TTr(5,4), TTdx(5,4);
  TTx_ref.setTTranks(2);
  randomize(TTx_ref);
  apply(TTOpA, TTx_ref, TTb);

  TTx.setOnes();

  copy(TTx, TTdx);
  double initialError = axpby(-1., TTx_ref, 1., TTdx);
  apply(TTOpA, TTx, TTr);
  double initialResidualNorm = axpby(-1., TTb, 1., TTr);


  double residualNorm = solveMALS(TTOpA, true, MALS_projection::RitzGalerkin, TTb, TTx, 1, eps, 10, 2, 1, 0, false, true);


  apply(TTOpA, TTx, TTr);
  double residualNorm_ref = axpby(-1., TTb, 1., TTr);
  EXPECT_NEAR(residualNorm_ref, residualNorm, eps*initialResidualNorm);

  std::cout << "initialResidualNorm: " << initialResidualNorm << ", newResidualNorm: " << residualNorm << "\n";

  copy(TTx, TTdx);
  double error = axpby(-1., TTx_ref, 1., TTdx);
  std::cout << "initialError: " << initialError << ", newError: " << error << "\n";
  EXPECT_NEAR(0, error/initialError, 0.01);
}
