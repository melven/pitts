#include <gtest/gtest.h>
#include "pitts_gmres.hpp"
#include "pitts_tensortrain_operator_apply_dense.hpp"
#include "pitts_tensortrain_operator_apply_transposed_op.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_dot.hpp"
#include "pitts_multivector_norm.hpp"
#include "pitts_multivector_scale.hpp"
#include "pitts_multivector_axpby.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"

namespace
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  using arr = Eigen::ArrayXd;
  constexpr auto eps = 1.e-10;

  void zero(MultiVector_double& mv)
  {
    for(int j = 0; j < mv.cols(); j++)
      for(int iChunk = 0; iChunk < mv.rowChunks(); iChunk++)
        mv.chunk(iChunk,j) = PITTS::Chunk<double>{};
  }
}

TEST(PITTS_GMRES, TTOp_dense_eye)
{
  TensorTrainOperator_double OpA(3, 5, 5);
  OpA.setEye();

  MultiVector_double b(5*5*5,3), x(5*5*5,3);
  zero(x);
  randomize(b);

  const auto [resNorm, relResNorm] = GMRES<arr>(OpA, true, b, x, 1, arr::Constant(3, 1.e-8), arr::Constant(3, 1.e-8), "TEST: ", true);

  EXPECT_NEAR(arr::Zero(3), resNorm, eps);
  EXPECT_NEAR(ConstEigenMap(b), ConstEigenMap(x), eps);
}

TEST(PITTS_GMRES, TTOp_dense_random_single_system_symmetric)
{
  TensorTrainOperator_double OpA(3, 5, 5);
  {
    TensorTrainOperator_double tmpA(3, 5, 5);
    tmpA.setTTranks(2);
    randomize(tmpA);
    normalize(tmpA);

    applyT(tmpA, tmpA, OpA);
    tmpA.setEye();
    const double nrmI = normalize(tmpA);
    axpby(nrmI, tmpA, 0.2*nrmI, OpA);
  }

  MultiVector_double b(5*5*5,1), x(5*5*5,1), r(5*5*5,1);
  randomize(x);
  randomize(b);

  const auto [resNorm, relResNorm] = GMRES<arr>(OpA, true, b, x, 50, arr::Constant(1, 1.e-4), arr::Constant(1, 1.e-8), "TEST: ", true);

  apply(OpA, x, r);
  const arr resNorm_ref = axpy_norm2(arr(arr::Constant(1,-1)), b, r);

  EXPECT_NEAR(resNorm_ref, resNorm, eps);
  ASSERT_EQ(1, resNorm.size());
  EXPECT_LE(resNorm(0), 1.e-4);
}

TEST(PITTS_GMRES, TTOp_dense_random_single_system)
{
  TensorTrainOperator_double OpA(3, 5, 5);
  OpA.setTTranks(2);
  randomize(OpA);
  normalize(OpA);

  TensorTrainOperator_double OpI(3, 5, 5);
  OpI.setEye();
  const double nrmI = normalize(OpI);
  axpby(nrmI, OpI, 0.2*nrmI, OpA);

  MultiVector_double b(5*5*5,1), x(5*5*5,1), r(5*5*5,1);
  randomize(x);
  randomize(b);

  const auto [resNorm, relResNorm] = GMRES<arr>(OpA, false, b, x, 50, arr::Constant(1, 1.e-4), arr::Constant(1, 1.e-8), "TEST: ", true);

  apply(OpA, x, r);
  const arr resNorm_ref = axpy_norm2(arr(arr::Constant(1,-1)), b, r);

  EXPECT_NEAR(resNorm_ref, resNorm, eps);
  ASSERT_EQ(1, resNorm.size());
  EXPECT_LE(resNorm(0), 1.e-4);
}
