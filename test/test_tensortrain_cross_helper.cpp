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

  const NestedIndices sized(4);
  EXPECT_EQ(4, sized.size());
  EXPECT_EQ(4, sized.link.size());
  EXPECT_EQ(4, sized.idx.size());
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


// ---- interface matrices (partial contractions on nested index sets) ----

#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_sweep_data.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include <random>

using PITTS::internal::tt_cross::interface_loop_from_left;
using PITTS::internal::tt_cross::interface_loop_from_right;
using PITTS::internal::defineSweepData;

namespace
{
  // reference: evaluate a tensor train at one multi-index by multiplying the core slices
  template<typename T>
  T evaluate(const PITTS::TensorTrain<T>& TT, const Eigen::RowVectorXi& multiIndex)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    mat v = mat::Identity(1, 1);
    for(int k = 0; k < multiIndex.size(); k++)
    {
      const auto& X = TT.subTensor(k);
      mat slice(X.r1(), X.r2());
      for(int i = 0; i < X.r1(); i++)
        for(int j = 0; j < X.r2(); j++)
          slice(i, j) = X(i, multiIndex(k), j);
      v = v * slice;
    }
    return v(0, 0);
  }

  // random nested left index sets I[k] of the given sizes for the given dimensions
  std::vector<NestedIndices> randomLeftIndexSets(const std::vector<int>& dims, const std::vector<int>& sizes, std::mt19937& gen)
  {
    std::vector<NestedIndices> I(sizes.size());
    for(int k = 0; k < sizes.size(); k++)
    {
      const int parentSize = (k == 0) ? 1 : sizes[k-1];
      std::uniform_int_distribution<int> linkDist(0, parentSize-1), idxDist(0, dims[k]-1);
      I[k] = NestedIndices(sizes[k]);
      for(int a = 0; a < sizes[k]; a++)
      {
        I[k].link[a] = linkDist(gen);
        I[k].idx[a] = idxDist(gen);
      }
    }
    return I;
  }

  // random nested right index sets J[k] for k = 1..d-1 (J[0] unused), |J_k| = sizes[k]
  std::vector<NestedIndices> randomRightIndexSets(const std::vector<int>& dims, const std::vector<int>& sizes, std::mt19937& gen)
  {
    const int nDim = dims.size();
    std::vector<NestedIndices> J(nDim);
    for(int k = nDim-1; k >= 1; k--)
    {
      const int childSize = (k == nDim-1) ? 1 : sizes[k+1];
      std::uniform_int_distribution<int> linkDist(0, childSize-1), idxDist(0, dims[k]-1);
      J[k] = NestedIndices(sizes[k]);
      for(int b = 0; b < sizes[k]; b++)
      {
        J[k].link[b] = linkDist(gen);
        J[k].idx[b] = idxDist(gen);
      }
    }
    return J;
  }
}

template<typename T>
class PITTS_TensorTrain_cross_helper_interfaces : public ::testing::Test
{
  public:
    using Type = T;
};

using InterfaceTestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PITTS_TensorTrain_cross_helper_interfaces, InterfaceTestTypes);

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, boundary_cores)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  constexpr auto eps = 1.e-5;

  const std::vector<int> dims = {4, 3, 5};
  TensorTrain TT(dims, 2);
  randomize(TT);

  std::vector<NestedIndices> I(2), J(3);
  I[0].link = {0, 0, 0};  I[0].idx = {3, 0, 3};       // duplicates are allowed
  J[2].link = {0, 0};     J[2].idx = {4, 1};

  auto Lx = defineSweepData<PITTS::Tensor2<Type>>(3, interface_loop_from_left<Type>(TT, I), interface_loop_from_right<Type>(TT, J));
  Lx.update(0, 2);

  // L_0(a,:) = X_0(0, idx[a], :)
  const auto& L0 = Lx.subTensor(0);
  ASSERT_EQ(3, L0.r1());
  ASSERT_EQ(2, L0.r2());
  for(int a = 0; a < 3; a++)
    for(int j = 0; j < 2; j++)
      EXPECT_NEAR(TT.subTensor(0)(0, I[0].idx[a], j), L0(a, j), eps);

  // R_2(:,b) = X_2(:, idx[b], 0)
  const auto& R2 = Lx.subTensor(2);
  ASSERT_EQ(2, R2.r1());
  ASSERT_EQ(2, R2.r2());
  for(int b = 0; b < 2; b++)
    for(int i = 0; i < 2; i++)
      EXPECT_NEAR(TT.subTensor(2)(i, J[2].idx[b], 0), R2(i, b), eps);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, random_TT_random_index_sets)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  constexpr auto eps = 1.e-4;

  const std::vector<int> dims = {3, 4, 2, 5, 3};
  const int nDim = dims.size();
  TensorTrain TT(dims);
  TT.setTTranks({2, 3, 2, 4});
  randomize(TT);

  // index set sizes deliberately differ from the TT ranks (no square assumptions anywhere)
  std::mt19937 gen(42);
  const std::vector<int> leftSizes = {3, 2, 5, 1};
  const std::vector<int> rightSizes = {0, 4, 2, 3, 2};
  const auto I = randomLeftIndexSets(dims, leftSizes, gen);
  const auto J = randomRightIndexSets(dims, rightSizes, gen);

  auto Lx = defineSweepData<PITTS::Tensor2<Type>>(nDim, interface_loop_from_left<Type>(TT, I), interface_loop_from_right<Type>(TT, J));

  // sweep over the bonds like the algorithm does: SweepData stores left and right results in the same
  // array, so only L_0..L_k and R_{k+1}..R_{d-1} can be valid at the same time (never overlapping ranges)
  for(int k = 0; k < nDim-1; k++)
  {
    Lx.update(k, k+1);
    const auto& L = Lx.subTensor(k);
    ASSERT_EQ(leftSizes[k], L.r1());
    ASSERT_EQ(TT.subTensor(k).r2(), L.r2());

    // check L_k and R_{k+1} through their product: (L_k R_{k+1})(a,b) = x(I_k[a], J_{k+1}[b])
    const Eigen::MatrixXi Ik = expandLeft(I, k);
    const Eigen::MatrixXi Jk1 = expandRight(J, k+1, nDim);
    const auto& R = Lx.subTensor(k+1);
    ASSERT_EQ(TT.subTensor(k+1).r1(), R.r1());
    ASSERT_EQ(rightSizes[k+1], R.r2());

    const mat LR = ConstEigenMap(L) * ConstEigenMap(R);
    for(int a = 0; a < Ik.rows(); a++)
      for(int b = 0; b < Jk1.rows(); b++)
      {
        Eigen::RowVectorXi multiIndex(nDim);
        multiIndex << Ik.row(a), Jk1.row(b);
        EXPECT_NEAR(evaluate(TT, multiIndex), LR(a, b), eps);
      }
  }
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, incremental_update_after_index_set_change)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  constexpr auto eps = 1.e-5;

  const std::vector<int> dims = {3, 3, 3, 3};
  const int nDim = dims.size();
  TensorTrain TT(dims, 2);
  randomize(TT);

  std::mt19937 gen(7);
  std::vector<NestedIndices> I = randomLeftIndexSets(dims, {2, 2, 2}, gen);
  const auto J = randomRightIndexSets(dims, {0, 2, 2, 2}, gen);

  auto Lx = defineSweepData<PITTS::Tensor2<Type>>(nDim, interface_loop_from_left<Type>(TT, I), interface_loop_from_right<Type>(TT, J));
  Lx.update(2, 3);

  // change I_1 (which makes I_2 point to new parents): after invalidate + update, L_1 and L_2 follow the new sets
  I[1] = randomLeftIndexSets(dims, {2, 3}, gen)[1];
  Lx.invalidate(1);
  Lx.update(2, 3);

  const Eigen::MatrixXi I2 = expandLeft(I, 2);
  const auto& L2 = Lx.subTensor(2);
  ASSERT_EQ(2, L2.r1());
  // check the recomputed L_2 through the product with R_3
  const Eigen::MatrixXi J3 = expandRight(J, 3, nDim);
  const auto LR = ConstEigenMap(L2) * ConstEigenMap(Lx.subTensor(3));
  for(int a = 0; a < I2.rows(); a++)
    for(int b = 0; b < J3.rows(); b++)
    {
      Eigen::RowVectorXi multiIndex(nDim);
      multiIndex << I2.row(a), J3.row(b);
      EXPECT_NEAR(evaluate(TT, multiIndex), LR(a, b), eps);
    }
}
