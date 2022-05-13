#include <gtest/gtest.h>
#include "pitts_gmres.hpp"
#include "pitts_tensortrain_operator_apply_dense.hpp"
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
  using arr = Eigen::Array<double, 1, Eigen::Dynamic>;
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

  const arr resNorm = GMRES<arr>(OpA, true, b, x, 1, arr::Constant(3, 1.e-8), arr::Constant(3, 1.e-8), "TEST: ", true);

  EXPECT_NEAR(arr::Zero(3), resNorm, eps);
  EXPECT_NEAR(ConstEigenMap(b), ConstEigenMap(x), eps);
}
