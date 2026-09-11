// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_cross_helper.hpp"
#include "pitts_tensor3_unfold.hpp"
#include <stdexcept>
#include <vector>

using PITTS::internal::tt_cross::NestedIndices;
using PITTS::internal::tt_cross::leftIndicesFromRows;
using PITTS::internal::tt_cross::rightIndicesFromColumns;
using PITTS::internal::tt_cross::expandLeft;
using PITTS::internal::tt_cross::expandRight;
using MultiIndexList = std::vector<std::vector<int>>;


TEST(PITTS_TensorTrain_cross_helper, NestedIndices_size_and_row_column_numbers)
{
  NestedIndices I;
  I.link = {0, 1, 0};
  I.idx = {2, 0, 1};
  ASSERT_EQ(3, I.size());

  // left unfolding with r_prev = 2 rows per mode index: row = link + idx*r_prev
  EXPECT_EQ((std::vector<int>{4, 1, 2}), I.rowIndices(2));

  // right unfolding with mode size n = 3: column = idx + link*n
  EXPECT_EQ((std::vector<int>{2, 3, 1}), I.columnIndices(3));

  NestedIndices empty;
  EXPECT_EQ(0, empty.size());
  EXPECT_TRUE(empty.rowIndices(5).empty());
  EXPECT_TRUE(empty.columnIndices(5).empty());
}

TEST(PITTS_TensorTrain_cross_helper, leftIndicesFromRows_roundtrip)
{
  const std::vector<int> rows = {4, 1, 2, 5};
  const NestedIndices I = leftIndicesFromRows(rows, 2);
  EXPECT_EQ((std::vector<int>{0, 1, 0, 1}), I.link);
  EXPECT_EQ((std::vector<int>{2, 0, 1, 2}), I.idx);
  EXPECT_EQ(rows, I.rowIndices(2));

  // r_prev = 1: the first bond, all links refer to the single empty multi-index
  const NestedIndices I0 = leftIndicesFromRows({3, 0}, 1);
  EXPECT_EQ((std::vector<int>{0, 0}), I0.link);
  EXPECT_EQ((std::vector<int>{3, 0}), I0.idx);

  EXPECT_THROW(leftIndicesFromRows({-1}, 2), std::invalid_argument);
  EXPECT_THROW(leftIndicesFromRows({0}, 0), std::invalid_argument);
}

TEST(PITTS_TensorTrain_cross_helper, rightIndicesFromColumns_roundtrip)
{
  const std::vector<int> columns = {2, 3, 1, 5};
  const NestedIndices J = rightIndicesFromColumns(columns, 3);
  EXPECT_EQ((std::vector<int>{2, 0, 1, 2}), J.idx);
  EXPECT_EQ((std::vector<int>{0, 1, 0, 1}), J.link);
  EXPECT_EQ(columns, J.columnIndices(3));

  EXPECT_THROW(rightIndicesFromColumns({-1}, 3), std::invalid_argument);
  EXPECT_THROW(rightIndicesFromColumns({0}, 0), std::invalid_argument);
}

TEST(PITTS_TensorTrain_cross_helper, NestedIndices_consistent_with_Tensor3_unfoldings)
{
  // the row/column numbering must match unfold_left / unfold_right of a TT core
  PITTS::Tensor3<double> t3(2, 3, 4);
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 4; k++)
        t3(i, j, k) = 100*i + 10*j + k;

  NestedIndices I;
  I.link = {1, 0, 1};
  I.idx = {2, 1, 0};
  const auto rows = I.rowIndices(t3.r1());
  const auto t2left = unfold_left(t3);
  for(int a = 0; a < I.size(); a++)
    for(int k = 0; k < 4; k++)
      EXPECT_EQ(t3(I.link[a], I.idx[a], k), t2left(rows[a], k));

  NestedIndices J;
  J.idx = {2, 1, 0, 2};
  J.link = {3, 0, 1, 1};
  const auto columns = J.columnIndices(t3.n());
  const auto t2right = unfold_right(t3);
  for(int b = 0; b < J.size(); b++)
    for(int i = 0; i < 2; i++)
      EXPECT_EQ(t3(i, J.idx[b], J.link[b]), t2right(i, columns[b]));
}

TEST(PITTS_TensorTrain_cross_helper, expandLeft)
{
  // d = 3 dimensions, left sets for bonds 0 and 1
  std::vector<NestedIndices> I(2);
  I[0].link = {0, 0};        // links into I_{-1} = {()}
  I[0].idx = {2, 0};
  I[1].link = {1, 0, 1};     // links into I_0
  I[1].idx = {0, 2, 1};

  EXPECT_EQ((MultiIndexList{{}}), expandLeft(I, -1));
  EXPECT_EQ((MultiIndexList{{2}, {0}}), expandLeft(I, 0));
  EXPECT_EQ((MultiIndexList{{0, 0}, {2, 2}, {0, 1}}), expandLeft(I, 1));
}

TEST(PITTS_TensorTrain_cross_helper, expandRight)
{
  // d = 3 dimensions, right sets J_1 (bond 0) and J_2 (bond 1); J[0] is unused
  const int nDim = 3;
  std::vector<NestedIndices> J(nDim);
  J[2].idx = {1, 0};
  J[2].link = {0, 0};        // links into J_3 = {()}
  J[1].idx = {2, 0, 1};
  J[1].link = {1, 1, 0};     // links into J_2

  EXPECT_EQ((MultiIndexList{{}}), expandRight(J, nDim, nDim));
  EXPECT_EQ((MultiIndexList{{1}, {0}}), expandRight(J, 2, nDim));
  EXPECT_EQ((MultiIndexList{{2, 0}, {0, 0}, {1, 1}}), expandRight(J, 1, nDim));
}
