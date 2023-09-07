#include <gtest/gtest.h>
#include "pitts_tensortrain_solve_mals_helper.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_random.hpp"
#include "eigen_test_helper.hpp"

namespace
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  using mat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;
  constexpr auto eps = 1.e-7;

  using namespace PITTS::internal::solve_mals;
}

TEST(PITTS_TensorTrain_solve_mals_helper, axpby_loop_from_right_nDim1)
{
  using ResultType = std::pair<Tensor3_double,Tensor2_double>;
  TensorTrain_double TTx(1,5), TTy(1,5);
  TTx.setUnit({0});
  TTy.setUnit({1});

  const auto loop = axpby_loop_from_right<double>(TTx, TTy);
  ResultType QB;
  loop(0, std::nullopt, QB);

  auto mapQ = ConstEigenMap(unfold_right(QB.first));
  EXPECT_NEAR(mat::Identity(2,2), mapQ * mapQ.transpose(), eps);
  mat Mleft(1, 2);
  Mleft << 1, -1;
  mat x_y_ref(1,5);
  x_y_ref << 1,-1,0,0,0;
  mat tmpB(Mleft.cols(), mapQ.rows());
  const auto& mapB = ConstEigenMap(QB.second);
  tmpB << mat::Identity(tmpB.rows()-mapB.rows(), tmpB.cols()), mapB;
  EXPECT_NEAR(x_y_ref, Mleft * tmpB * mapQ, eps);
}

TEST(PITTS_TensorTrain_solve_mals_helper, axpby_loop_from_right_nDim2_rank1)
{
  using ResultType = std::pair<Tensor3_double,Tensor2_double>;
  TensorTrain_double TTx(2,5), TTy(2,5);
  TTx.setUnit({0,0});
  TTy.setUnit({1,2});

  const auto loop = axpby_loop_from_right<double>(TTx, TTy);
  std::vector<ResultType> QB(2);
  loop(1, std::nullopt, QB[1]);
  loop(0, QB[1], QB[0]);

  mat Mleft(1, 2);
  Mleft << 1, -1;

  Tensor2_double newSubT0(1,5*2);
#ifndef PITTS_TENSORTRAIN_PLAIN_AXPBY
  EigenMap(newSubT0) = Mleft * ConstEigenMap(unfold_right(QB[0].first));
#else
  EigenMap(newSubT0) = Mleft * ConstEigenMap(QB[0].second) * ConstEigenMap(unfold_right(QB[0].first));
#endif
  fold_right(newSubT0, 5, QB[0].first);

  std::vector<Tensor3_double> tmpXY(QB.size());
  for(int i = 0; i < QB.size(); i++)
    tmpXY[i] = std::move(QB[i].first);

  TensorTrain_double TTxy(std::move(tmpXY));

  TensorTrain_double TTxy_ref(TTy);
  double xy_nrm = axpby(1., TTx, -1., TTxy_ref);
  const double error = axpby(-1., TTxy, xy_nrm, TTxy_ref);
  EXPECT_NEAR(0, error, eps);
}
