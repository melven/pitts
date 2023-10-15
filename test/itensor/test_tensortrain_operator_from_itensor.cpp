// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

// workaround ITensor / Eigen LAPACK definition problems
#ifdef EIGEN_USE_LAPACKE
#undef EIGEN_USE_LAPACKE
#endif

#include "pitts_tensortrain_operator_from_itensor.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include <itensor/all.h>
#include <iostream>


TEST(PITTS_TensorTrainOperator_fromITensor, 2x2_zero)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  constexpr auto eps = 1.e-10;

  const int N = 2;
  const auto sites = itensor::SpinHalf(N,{"ConserveQNs=",false});

  const auto ampo = itensor::AutoMPO(sites);

  const auto mpo = itensor::toMPO(ampo);
  std::cout << "mpo:\n" << mpo;

  const TensorTrainOperator_double ttOp = PITTS::fromITensor<double>(mpo);

  const std::vector<int> dimsRef = {2,2};
  ASSERT_EQ(dimsRef, ttOp.column_dimensions());
  ASSERT_EQ(dimsRef, ttOp.row_dimensions());

  //const std::vector<int> ranksRef = {2};
  //ASSERT_EQ(ranksRef, ttOp.getTTranks());

  ASSERT_NEAR(0., norm2(ttOp.tensorTrain()), eps);
}


TEST(PITTS_TensorTrainOperator_fromITensor, 2x2_unit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  constexpr auto eps = 1.e-10;

  const int N = 2;
  const auto sites = itensor::SpinHalf(N,{"ConserveQNs=",false});

  auto ampo = itensor::AutoMPO(sites);
  ampo += "projUp",1;
  ampo += "projDn",1;
  // the default for single core operators is that all other cores stay identical...

  auto mpo = itensor::toMPO(ampo);
  std::cout << "mpo:\n" << mpo;

  const TensorTrainOperator_double ttOp = PITTS::fromITensor<double>(mpo);

  const std::vector<int> dimsRef = {2,2};
  ASSERT_EQ(dimsRef, ttOp.column_dimensions());
  ASSERT_EQ(dimsRef, ttOp.row_dimensions());

  //const std::vector<int> ranksRef = {2};
  //ASSERT_EQ(ranksRef, ttOp.getTTranks());

  // operator should be the identity
  TensorTrainOperator_double ttOp_ref(dimsRef, dimsRef);
  ttOp_ref.setEye();

  /*
  std::cout << "TTOp subT(0):\n";
  for(int j = 0; j < ttOp.getTTranks()[0]; j++)
  {
    for(int i = 0; i < 4; i++)
      std::cout << " " << ttOp.tensorTrain().subTensor(0)(0,i,j);
    std::cout << "\n";
  }
  std::cout << "TTOp subT(1):\n";
  for(int j = 0; j < ttOp.getTTranks()[0]; j++)
  {
    for(int i = 0; i < 4; i++)
      std::cout << " " << ttOp.tensorTrain().subTensor(1)(j,i,0);
    std::cout << "\n";
  }
  */

  ASSERT_NEAR(0., axpby(1., ttOp.tensorTrain(), -1., ttOp_ref.tensorTrain()), eps);
}
