// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_cross_helper.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "eigen_test_helper.hpp"
#include <stdexcept>
#include <vector>

using PITTS::internal::tt_cross::NestedIndices;
using PITTS::internal::tt_cross::leftIndicesFromRows;
using PITTS::internal::tt_cross::rightIndicesFromColumns;
using PITTS::internal::tt_cross::expandLeft;
using PITTS::internal::tt_cross::expandRight;

namespace
{
  // helper: build an integer matrix from an initializer list of rows
  Eigen::MatrixXi rows(std::initializer_list<std::vector<int>> data)
  {
    const int nRows = data.size();
    const int nCols = nRows > 0 ? data.begin()->size() : 0;
    Eigen::MatrixXi result(nRows, nCols);
    int i = 0;
    for(const auto& row: data)
    {
      for(int j = 0; j < nCols; j++)
        result(i, j) = row[j];
      i++;
    }
    return result;
  }
}


TEST(PITTS_TensorTrain_cross_helper, NestedIndices_size)
{
  NestedIndices I;
  I.link = {0, 1, 0};
  I.idx = {2, 0, 1};
  EXPECT_EQ(3, I.size());

  NestedIndices empty;
  EXPECT_EQ(0, empty.size());
}

TEST(PITTS_TensorTrain_cross_helper, leftIndicesFromRows)
{
  const NestedIndices I = leftIndicesFromRows({4, 1, 2, 5}, 2);
  EXPECT_EQ((std::vector<int>{0, 1, 0, 1}), I.link);
  EXPECT_EQ((std::vector<int>{2, 0, 1, 2}), I.idx);

  // r_prev = 1: the first bond, all links refer to the single empty multi-index
  const NestedIndices I0 = leftIndicesFromRows({3, 0}, 1);
  EXPECT_EQ((std::vector<int>{0, 0}), I0.link);
  EXPECT_EQ((std::vector<int>{3, 0}), I0.idx);

  EXPECT_THROW(leftIndicesFromRows({-1}, 2), std::invalid_argument);
  EXPECT_THROW(leftIndicesFromRows({0}, 0), std::invalid_argument);
}

TEST(PITTS_TensorTrain_cross_helper, rightIndicesFromColumns)
{
  const NestedIndices J = rightIndicesFromColumns({2, 3, 1, 5}, 3);
  EXPECT_EQ((std::vector<int>{2, 0, 1, 2}), J.idx);
  EXPECT_EQ((std::vector<int>{0, 1, 0, 1}), J.link);

  EXPECT_THROW(rightIndicesFromColumns({-1}, 3), std::invalid_argument);
  EXPECT_THROW(rightIndicesFromColumns({0}, 0), std::invalid_argument);
}

TEST(PITTS_TensorTrain_cross_helper, decoding_consistent_with_Tensor3_unfoldings)
{
  // decoding a row/column number must give the (link, idx) that address the same entries of the Tensor3
  PITTS::Tensor3<double> t3(2, 3, 4);
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 4; k++)
        t3(i, j, k) = 100*i + 10*j + k;

  const auto t2left = unfold_left(t3);
  for(int row = 0; row < t2left.r1(); row++)
  {
    const NestedIndices I = leftIndicesFromRows({row}, t3.r1());
    for(int k = 0; k < 4; k++)
      EXPECT_EQ(t3(I.link[0], I.idx[0], k), t2left(row, k));
  }

  const auto t2right = unfold_right(t3);
  for(int col = 0; col < t2right.r2(); col++)
  {
    const NestedIndices J = rightIndicesFromColumns({col}, t3.n());
    for(int i = 0; i < 2; i++)
      EXPECT_EQ(t3(i, J.idx[0], J.link[0]), t2right(i, col));
  }
}

TEST(PITTS_TensorTrain_cross_helper, expandLeft)
{
  // d = 3 dimensions, left sets for bonds 0 and 1
  std::vector<NestedIndices> I(2);
  I[0].link = {0, 0};        // links into I_{-1} = {()}
  I[0].idx = {2, 0};
  I[1].link = {1, 0, 1};     // links into I_0
  I[1].idx = {0, 2, 1};

  const Eigen::MatrixXi boundary = expandLeft(I, -1);
  EXPECT_EQ(1, boundary.rows());
  EXPECT_EQ(0, boundary.cols());
  EXPECT_NEAR(rows({{2}, {0}}), expandLeft(I, 0), 0);
  EXPECT_NEAR(rows({{0, 0}, {2, 2}, {0, 1}}), expandLeft(I, 1), 0);
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

  const Eigen::MatrixXi boundary = expandRight(J, nDim, nDim);
  EXPECT_EQ(1, boundary.rows());
  EXPECT_EQ(0, boundary.cols());
  EXPECT_NEAR(rows({{1}, {0}}), expandRight(J, 2, nDim), 0);
  EXPECT_NEAR(rows({{2, 0}, {0, 0}, {1, 1}}), expandRight(J, 1, nDim), 0);
}
