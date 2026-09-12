// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_sweep_data.hpp"
#include <vector>
#include <utility>

using PITTS::internal::defineSweepData;
using PITTS::internal::optional_cref;

namespace
{
  // record which dimensions were computed in which order, and from which side
  using CallLog = std::vector<std::pair<char, int>>;

  // the result of dimension iDim is the "prefix sum" iDim+1 + (result of the previous dimension)
  auto prefixSumFromLeft(CallLog& log)
  {
    return [&log](int iDim, optional_cref<int> prev, int& result)
    {
      log.emplace_back('L', iDim);
      result = (iDim+1) + (prev ? prev->get() : 0);
    };
  }

  auto prefixSumFromRight(CallLog& log)
  {
    return [&log](int iDim, optional_cref<int> prev, int& result)
    {
      log.emplace_back('R', iDim);
      result = (iDim+1) + (prev ? prev->get() : 0);
    };
  }
}


TEST(PITTS_TensorTrain_sweep_data, initial_state)
{
  CallLog log;
  auto data = defineSweepData<int>(5, prefixSumFromLeft(log), prefixSumFromRight(log));

  EXPECT_EQ(5, data.nDim());
  EXPECT_EQ(-1, data.leftDim());
  EXPECT_EQ(5, data.rightDim());
  EXPECT_FALSE(data.left());
  EXPECT_FALSE(data.right());
  EXPECT_EQ(5, data.data().size());
  EXPECT_TRUE(log.empty());
}

TEST(PITTS_TensorTrain_sweep_data, update_computes_from_both_sides_in_order)
{
  CallLog log;
  auto data = defineSweepData<int>(5, prefixSumFromLeft(log), prefixSumFromRight(log));

  data.update(1, 3);
  EXPECT_EQ(1, data.leftDim());
  EXPECT_EQ(3, data.rightDim());
  EXPECT_EQ((CallLog{{'L', 0}, {'L', 1}, {'R', 4}, {'R', 3}}), log);

  // results: from the left 1, 1+2; from the right 5, 5+4
  ASSERT_TRUE(data.left());
  ASSERT_TRUE(data.right());
  EXPECT_EQ(3, data.left()->get());
  EXPECT_EQ(9, data.right()->get());
  EXPECT_EQ(1, data.subTensor(0));
  EXPECT_EQ(3, data.subTensor(1));
  EXPECT_EQ(9, data.subTensor(3));
  EXPECT_EQ(5, data.subTensor(4));
}

TEST(PITTS_TensorTrain_sweep_data, update_only_computes_missing_dimensions)
{
  CallLog log;
  auto data = defineSweepData<int>(5, prefixSumFromLeft(log), prefixSumFromRight(log));

  data.update(1, 3);
  log.clear();

  // extend to the right: only dimension 2 from the left
  data.update(2, 3);
  EXPECT_EQ((CallLog{{'L', 2}}), log);
  EXPECT_EQ(6, data.subTensor(2));
  log.clear();

  // same state again: nothing to do
  data.update(2, 3);
  EXPECT_TRUE(log.empty());

  // shrinking the valid range does not recompute anything (the old values stay, only the range changes)
  data.update(0, 4);
  EXPECT_TRUE(log.empty());
  EXPECT_EQ(0, data.leftDim());
  EXPECT_EQ(4, data.rightDim());
  EXPECT_EQ(1, data.left()->get());
  EXPECT_EQ(5, data.right()->get());
}

TEST(PITTS_TensorTrain_sweep_data, invalidate_and_recompute)
{
  CallLog log;
  auto data = defineSweepData<int>(5, prefixSumFromLeft(log), prefixSumFromRight(log));

  data.update(3, 4);
  log.clear();

  // dimension 2 changed: everything from dimension 2 on is invalid from the left, from 2 on (downwards) from the right
  data.invalidate(2);
  EXPECT_EQ(1, data.leftDim());
  EXPECT_EQ(4, data.rightDim());

  data.update(3, 4);
  EXPECT_EQ((CallLog{{'L', 2}, {'L', 3}}), log);
  log.clear();

  // a local problem of nMALS=2 dimensions starting at 1 invalidates 1..2 from the left and up to 2 from the right
  data.update(0, 2);
  log.clear();
  data.invalidate(1, 2);
  EXPECT_EQ(0, data.leftDim());
  EXPECT_EQ(3, data.rightDim());
  data.update(0, 2);
  EXPECT_EQ((CallLog{{'R', 2}}), log);

  // invalidating the last dimension drops everything computed from the right (R_3 depends on R_4)
  log.clear();
  data.invalidate(4);
  EXPECT_EQ(0, data.leftDim());
  EXPECT_EQ(5, data.rightDim());
  data.update(0, 2);
  EXPECT_EQ((CallLog{{'R', 4}, {'R', 3}, {'R', 2}}), log);

  // invalidating a dimension that is valid from neither side changes nothing
  log.clear();
  data.update(0, 3);
  data.invalidate(1);
  EXPECT_EQ(0, data.leftDim());
  EXPECT_EQ(3, data.rightDim());
  data.update(0, 3);
  EXPECT_TRUE(log.empty());
}

TEST(PITTS_TensorTrain_sweep_data, full_sweep_pattern)
{
  // typical use: left-to-right sweep keeps left()/right() around the current core
  CallLog log;
  const int nDim = 4;
  auto data = defineSweepData<int>(nDim, prefixSumFromLeft(log), prefixSumFromRight(log));

  for(int iDim = 0; iDim < nDim; iDim++)
  {
    data.update(iDim-1, iDim+1);
    EXPECT_EQ(iDim-1, data.leftDim());
    EXPECT_EQ(iDim+1, data.rightDim());
    EXPECT_EQ(iDim > 0, static_cast<bool>(data.left()));
    EXPECT_EQ(iDim+1 < nDim, static_cast<bool>(data.right()));
  }
  // right side computed once completely at the start, left side incrementally
  EXPECT_EQ((CallLog{{'R', 3}, {'R', 2}, {'R', 1}, {'L', 0}, {'L', 1}, {'L', 2}}), log);
}
