#include <gtest/gtest.h>
#include "pitts_tensortrain_solve_mals.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_random.hpp"

namespace
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  using mat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
  constexpr auto eps = 1.e-10;
}

TEST(PITTS_TensorTrain_solve_mals, Opeye_ones)
{
  TensorTrainOperator_double TTOpA(1,5,5);
  TTOpA.setEye();
  TensorTrain_double TTx(1,5), TTb(1,5);
  TTb.setOnes();
  TTx.setOnes();

  double error = solveMALS(TTOpA, true, TTb, TTx, 1, eps, 10);
  ASSERT_NEAR(0, error, eps);

  double errNrm = axpby(-1., TTb, 1., TTx);
  ASSERT_NEAR(0, errNrm, eps);
}

