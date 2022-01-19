#include <gtest/gtest.h>
#include "pitts_tensortrain_operator_to_itensor.hpp"
#include "pitts_tensortrain_operator_from_itensor.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include <itensor/all.h>
#include <iostream>


TEST(PITTS_TensorTrainOperator_toITensor, 2x3_4x5_3x3_random_fromToITensor)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  constexpr auto eps = 1.e-10;

  const std::vector<int> row_dims = {2,4,3};
  const std::vector<int> col_dims = {3,5,3};
  const std::vector<int> ranks{2,3};

  auto ttOp = TensorTrainOperator_double(row_dims, col_dims);
  ttOp.setTTranks(ranks);

  randomize(ttOp);

  const auto mpo = toITensor(ttOp);

  ASSERT_EQ(row_dims.size(), length(mpo));

  const auto idx1 = siteInds(mpo, 1);
  ASSERT_EQ(2, length(idx1));
  ASSERT_EQ(row_dims[0], dim(idx1(1)));
  ASSERT_EQ(col_dims[0], dim(idx1(2)));

  const auto idx2 = siteInds(mpo, 2);
  ASSERT_EQ(2, length(idx2));
  ASSERT_EQ(row_dims[1], dim(idx2(1)));
  ASSERT_EQ(col_dims[1], dim(idx2(2)));

  const auto idx3 = siteInds(mpo, 3);
  ASSERT_EQ(2, length(idx3));
  ASSERT_EQ(row_dims[2], dim(idx3(1)));
  ASSERT_EQ(col_dims[2], dim(idx3(2)));

  ASSERT_EQ(ranks.size(), length(mpo)-1);
  ASSERT_EQ(ranks[0], dim(linkIndex(mpo, 1)));
  ASSERT_EQ(ranks[1], dim(linkIndex(mpo, 2)));

  std::cout << "mpo:\n" << mpo;
  auto ttOp_from_mpo = PITTS::fromITensor<double>(mpo);
  ASSERT_NEAR(0., axpby(1., ttOp.tensorTrain(), -1., ttOp_from_mpo.tensorTrain()), eps);
}

