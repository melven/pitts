// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_cross_helper.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "eigen_test_helper.hpp"
#include <algorithm>
#include <array>
#include <set>
#include <stdexcept>
#include <utility>
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

  // check that J is a valid right-nested family of sets for the given dimensions and sizes (links and indices in range, no duplicates)
  void checkRightNested(const std::vector<NestedIndices>& J, const std::vector<int>& dims, const std::vector<int>& sizes)
  {
    const int nDim = dims.size();
    ASSERT_EQ(nDim, J.size());
    EXPECT_EQ(0, J[0].size());
    for(int k = 1; k < nDim; k++)
    {
      ASSERT_EQ(sizes[k], J[k].size());
      const int childSize = (k == nDim-1) ? 1 : J[k+1].size();
      std::set<std::pair<int,int>> unique;
      for(int b = 0; b < J[k].size(); b++)
      {
        EXPECT_LE(0, J[k].link[b]);
        EXPECT_LT(J[k].link[b], childSize);
        EXPECT_LE(0, J[k].idx[b]);
        EXPECT_LT(J[k].idx[b], dims[k]);
        unique.emplace(J[k].link[b], J[k].idx[b]);
      }
      EXPECT_EQ(J[k].size(), unique.size());
    }
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

TEST(PITTS_TensorTrain_cross_helper, randomRightIndexSets)
{
  using PITTS::internal::tt_cross::randomRightIndexSets;

  const std::vector<int> dims = {2, 5, 3, 2};
  const int nDim = dims.size();
  const int size = 4;
  const std::vector<NestedIndices> J = randomRightIndexSets(dims, size);

  // the sets are valid and nested; their sizes are limited by the number of candidates |S_k x J_{k+1}|
  checkRightNested(J, dims, {0, 4, 4, 2});

  for(int k = 1; k < nDim; k++)
  {
    // the full multi-indices are consistent
    const Eigen::MatrixXi full = expandRight(J, k, nDim);
    EXPECT_EQ(J[k].size(), full.rows());
    EXPECT_EQ(nDim-k, full.cols());
  }

  EXPECT_THROW(randomRightIndexSets(dims, 0), std::invalid_argument);
}


// ---- partial contractions on nested index sets and the interface matrices selected from them ----

#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensor3_random.hpp"
#include "pitts_tensortrain_sweep_data.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include <random>

using PITTS::internal::tt_cross::contraction_loop_from_left;
using PITTS::internal::tt_cross::contraction_loop_from_right;
using PITTS::internal::tt_cross::interfaceLeft;
using PITTS::internal::tt_cross::interfaceRight;
using PITTS::internal::defineSweepData;

namespace
{
  // evaluate a tensor train at one multi-index: the scalar product with the corresponding unit tensor
  template<typename T>
  T evaluate(const PITTS::TensorTrain<T>& TT, const Eigen::RowVectorXi& multiIndex)
  {
    PITTS::TensorTrain<T> unit(TT.dimensions());
    unit.setUnit(std::vector<int>(multiIndex.begin(), multiIndex.end()));
    return dot(unit, TT);
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

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, interfaceLeft_interfaceRight)
{
  using Type = typename TestFixture::Type;
  constexpr auto eps = 1.e-6;

  PITTS::Tensor3<Type> LX(2, 3, 4);
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int l = 0; l < 4; l++)
        LX(i, j, l) = Type(i + 10*j + 100*l);

  NestedIndices I;
  I.link = {1, 0, 1};  I.idx = {2, 0, 2};   // duplicates are allowed
  const PITTS::Tensor2<Type> L = interfaceLeft(LX, I);
  ASSERT_EQ(3, L.r1());
  ASSERT_EQ(4, L.r2());
  for(int a = 0; a < 3; a++)
    for(int l = 0; l < 4; l++)
      EXPECT_NEAR(Type(I.link[a] + 10*I.idx[a] + 100*l), L(a, l), eps);

  NestedIndices J;
  J.idx = {1, 1};  J.link = {3, 0};
  const PITTS::Tensor2<Type> R = interfaceRight(LX, J);
  ASSERT_EQ(2, R.r1());
  ASSERT_EQ(2, R.r2());
  for(int b = 0; b < 2; b++)
    for(int i = 0; i < 2; i++)
      EXPECT_NEAR(Type(i + 10*J.idx[b] + 100*J.link[b]), R(i, b), eps);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, contract_interface_boundary)
{
  using Type = typename TestFixture::Type;
  using PITTS::internal::tt_cross::contractLeftInterface;
  using PITTS::internal::tt_cross::contractRightInterface;
  using optional_cref_t3 = PITTS::internal::optional_cref<PITTS::Tensor3<Type>>;
  constexpr auto eps = 1.e-6;

  PITTS::Tensor3<Type> B(2, 3, 2), neighbor(2, 2, 2), C;
  randomize(B);
  randomize(neighbor);

  // at the boundary (nullopt) the interface is the 1x1 identity, so the block is passed through unchanged
  std::vector<NestedIndices> I(1), J(2);
  contractLeftInterface<Type>(std::nullopt, I, 0, B, C);
  EXPECT_NEAR(ConstEigenMap(unfold_left(B)), ConstEigenMap(unfold_left(C)), eps);
  contractRightInterface<Type>(B, std::nullopt, J, 1, C);
  EXPECT_NEAR(ConstEigenMap(unfold_left(B)), ConstEigenMap(unfold_left(C)), eps);

  // otherwise the interface of the neighboring contraction is contracted
  I[0].link = {0, 1, 1};  I[0].idx = {1, 0, 1};
  contractLeftInterface<Type>(optional_cref_t3(std::cref(neighbor)), I, 0, B, C);
  ASSERT_EQ(3, C.r1());
  PITTS::Tensor3<Type> Cref;
  PITTS::internal::normalize_contract1(interfaceLeft(neighbor, I[0]), B, Cref);
  EXPECT_NEAR(ConstEigenMap(unfold_left(Cref)), ConstEigenMap(unfold_left(C)), eps);

  J[1].idx = {0, 1};  J[1].link = {1, 0};
  contractRightInterface<Type>(B, optional_cref_t3(std::cref(neighbor)), J, 1, C);
  ASSERT_EQ(2, C.r2());
  PITTS::internal::normalize_contract2(B, interfaceRight(neighbor, J[1]), Cref);
  EXPECT_NEAR(ConstEigenMap(unfold_left(Cref)), ConstEigenMap(unfold_left(C)), eps);
}

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

  auto Lx = defineSweepData<PITTS::Tensor3<Type>>(3, contraction_loop_from_left<Type>(TT, I), contraction_loop_from_right<Type>(TT, J));
  Lx.update(0, 2);

  // LX_0 = X_0 (L_{-1} = 1), L_0(a,:) = X_0(0, idx[a], :)
  const auto& LX0 = Lx.subTensor(0);
  ASSERT_EQ(1, LX0.r1());
  ASSERT_EQ(4, LX0.n());
  ASSERT_EQ(2, LX0.r2());
  const PITTS::Tensor2<Type> L0 = interfaceLeft(LX0, I[0]);
  ASSERT_EQ(3, L0.r1());
  ASSERT_EQ(2, L0.r2());
  for(int a = 0; a < 3; a++)
    for(int j = 0; j < 2; j++)
      EXPECT_NEAR(TT.subTensor(0)(0, I[0].idx[a], j), L0(a, j), eps);

  // XR_2 = X_2 (R_3 = 1), R_2(:,b) = X_2(:, idx[b], 0)
  const auto& XR2 = Lx.subTensor(2);
  ASSERT_EQ(2, XR2.r1());
  ASSERT_EQ(5, XR2.n());
  ASSERT_EQ(1, XR2.r2());
  const PITTS::Tensor2<Type> R2 = interfaceRight(XR2, J[2]);
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

  auto Lx = defineSweepData<PITTS::Tensor3<Type>>(nDim, contraction_loop_from_left<Type>(TT, I), contraction_loop_from_right<Type>(TT, J));

  // sweep over the bonds like the algorithm does: SweepData stores left and right results in the same
  // array, so only LX_0..LX_k and XR_{k+1}..XR_{d-1} can be valid at the same time (never overlapping ranges)
  for(int k = 0; k < nDim-1; k++)
  {
    Lx.update(k, k+1);
    const auto& LX = Lx.subTensor(k);
    ASSERT_EQ(k == 0 ? 1 : leftSizes[k-1], LX.r1());
    ASSERT_EQ(dims[k], LX.n());
    ASSERT_EQ(TT.subTensor(k).r2(), LX.r2());
    const PITTS::Tensor2<Type> L = interfaceLeft(LX, I[k]);
    ASSERT_EQ(leftSizes[k], L.r1());

    // check L_k and R_{k+1} through their product: (L_k R_{k+1})(a,b) = x(I_k[a], J_{k+1}[b])
    const Eigen::MatrixXi Ik = expandLeft(I, k);
    const Eigen::MatrixXi Jk1 = expandRight(J, k+1, nDim);
    const auto& XR = Lx.subTensor(k+1);
    ASSERT_EQ(TT.subTensor(k+1).r1(), XR.r1());
    ASSERT_EQ(dims[k+1], XR.n());
    ASSERT_EQ(k+1 == nDim-1 ? 1 : rightSizes[k+2], XR.r2());
    const PITTS::Tensor2<Type> R = interfaceRight(XR, J[k+1]);
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

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, local_block_vs_direct_evaluation)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  using PITTS::internal::tt_cross::localBlockFromLeft;
  using PITTS::internal::tt_cross::localBlockFromRight;
  constexpr auto eps = 1.e-4;

  const std::vector<int> dims = {3, 4, 2, 5, 3};
  const int nDim = dims.size();
  TensorTrain TT(dims);
  TT.setTTranks({2, 3, 2, 4});
  randomize(TT);

  std::mt19937 gen(11);
  const std::vector<int> leftSizes = {2, 3, 2, 3};
  const std::vector<int> rightSizes = {0, 3, 2, 3, 2};
  const auto I = randomLeftIndexSets(dims, leftSizes, gen);
  const auto J = randomRightIndexSets(dims, rightSizes, gen);

  for(int nMALS: {1, 2})
  {
    auto Lx = defineSweepData<PITTS::Tensor3<Type>>(nDim, contraction_loop_from_left<Type>(TT, I), contraction_loop_from_right<Type>(TT, J));

    for(int k = 0; k + nMALS <= nDim; k++)
    {
      // left-to-right: the cached contraction of the first core of the block, interface from the right
      Lx.update(k, k+nMALS);
      PITTS::Tensor3<Type> B;
      localBlockFromLeft<Type>(Lx.left()->get(), TT, k, nMALS, k+nMALS < nDim ? Lx.right() : std::nullopt, J, B);

      const Eigen::MatrixXi Ik = expandLeft(I, k-1);
      const Eigen::MatrixXi Jk = expandRight(J, k+nMALS, nDim);
      ASSERT_EQ(Ik.rows(), B.r1());
      ASSERT_EQ(Jk.rows(), B.r2());
      ASSERT_EQ(nMALS == 1 ? dims[k] : dims[k]*dims[k+1], B.n());

      for(int a = 0; a < B.r1(); a++)
        for(int b = 0; b < B.r2(); b++)
          for(int j = 0; j < B.n(); j++)
          {
            // combine stores the modes of the block as j = i_k + i_{k+1}*n_k
            Eigen::RowVectorXi multiIndex(nDim);
            if( nMALS == 2 )
              multiIndex << Ik.row(a), j % dims[k], j / dims[k], Jk.row(b);
            else
              multiIndex << Ik.row(a), j % dims[k], Jk.row(b);
            EXPECT_NEAR(evaluate(TT, multiIndex), B(a, j, b), eps);
          }

      // right-to-left: the cached contraction of the last core of the block, interface from the left
      Lx.invalidate(0, nDim);
      Lx.update(k-1, k+nMALS-1);
      PITTS::Tensor3<Type> Bright;
      localBlockFromRight<Type>(k > 0 ? Lx.left() : std::nullopt, I, TT, k, nMALS, Lx.right()->get(), Bright);
      EXPECT_NEAR(ConstEigenMap(unfold_left(B)), ConstEigenMap(unfold_left(Bright)), eps);
      Lx.invalidate(0, nDim);
    }
  }
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, error_cross_contraction_vs_direct_evaluation)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  using PITTS::internal::tt_cross::contractLeftInterface;
  using PITTS::internal::tt_cross::contractRightInterface;
  using PITTS::internal::tt_cross::evaluateAt;
  constexpr auto eps = 1.e-4;

  // The error crosses of the driver mix two families of nested index sets (the left part may come from
  // the index sets of y, the right part from those of the error approximation z) and both families change
  // during a sweep. This checks the cached contractions used there against a direct evaluation.
  const std::vector<int> dims = {3, 4, 2, 5, 3};
  const int nDim = dims.size();
  TensorTrain TT(dims);
  TT.setTTranks({2, 3, 2, 4});
  randomize(TT);

  std::mt19937 gen(5);
  std::vector<NestedIndices> I = randomLeftIndexSets(dims, {2, 3, 2, 3}, gen);
  std::vector<NestedIndices> Jz = randomRightIndexSets(dims, {0, 3, 2, 3, 2}, gen);

  // block on the cross (I_{k-1}, S_k, Jz_{k+1}) resp. (I_{kLeft-1}, S_k, Jz_{k+1}) vs. direct evaluation
  const auto checkBlock = [&](const PITTS::Tensor3<Type>& block, const Eigen::MatrixXi& left, int firstDim,
                              const Eigen::MatrixXi& right)
  {
    ASSERT_EQ(left.rows(), block.r1());
    ASSERT_EQ(dims[firstDim], block.n());
    ASSERT_EQ(right.rows(), block.r2());

    Eigen::MatrixXi multiIndices(block.r1()*block.n()*block.r2(), nDim);
    int row = 0;
    for(int b = 0; b < block.r2(); b++)
      for(int j = 0; j < block.n(); j++)
        for(int a = 0; a < block.r1(); a++)
          multiIndices.row(row++) << left.row(a), j, right.row(b);

    PITTS::Tensor3<Type> reference;
    evaluateAt(TT, multiIndices, reference);
    row = 0;
    for(int b = 0; b < block.r2(); b++)
      for(int j = 0; j < block.n(); j++)
        for(int a = 0; a < block.r1(); a++, row++)
          EXPECT_NEAR(reference(row, 0, 0), block(a, j, b), eps);
  };

  // left-to-right: left interface from I, right interface from Jz, with I changing as in the driver
  auto L = defineSweepData<PITTS::Tensor3<Type>>(nDim, contraction_loop_from_left<Type>(TT, I), contraction_loop_from_right<Type>(TT, Jz));
  for(int k = 0; k < nDim-1; k++)
  {
    L.update(k, k+1);
    PITTS::Tensor3<Type> block;
    contractRightInterface<Type>(L.left()->get(), L.right(), Jz, k+1, block);
    checkBlock(block, expandLeft(I, k-1), k, expandRight(Jz, k+1, nDim));

    // the sweep selects a new index set I_k here, which invalidates the contraction of the next dimension
    I[k] = randomLeftIndexSets(dims, std::vector<int>(k+1, 2), gen)[k];
    L.invalidate(k+1);
  }

  // right-to-left: left interface from I, right interface from Jz, with Jz changing
  auto R = defineSweepData<PITTS::Tensor3<Type>>(nDim, contraction_loop_from_left<Type>(TT, I), contraction_loop_from_right<Type>(TT, Jz));
  for(int k = nDim-1; k > 0; k--)
  {
    R.update(k-1, k);
    PITTS::Tensor3<Type> block;
    contractLeftInterface<Type>(R.left(), I, k-1, R.right()->get(), block);
    checkBlock(block, expandLeft(I, k-1), k, expandRight(Jz, k+1, nDim));

    Jz[k] = randomRightIndexSets(dims, std::vector<int>(nDim, 2), gen)[k];
    R.invalidate(k-1);
  }
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, FunctionOnCrosses_two_inputs_vs_direct_evaluation)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  using PITTS::internal::tt_cross::randomIndices;
  constexpr auto eps = 1.e-4;

  // f(x^(1), x^(2)) of two tensor trains with different ranks; not symmetric, so the order of the arguments matters
  const std::vector<int> dims = {3, 4, 2, 5, 3};
  const int nDim = dims.size();
  TensorTrain TTa(dims), TTb(dims);
  TTa.setTTranks({2, 3, 2, 4});
  TTb.setTTranks({3, 1, 2, 2});
  randomize(TTa);
  randomize(TTb);
  const auto f = [](Type a, Type b){return a*b + Type(2)*b;};
  using Evaluator = PITTS::internal::tt_cross::FunctionOnCrosses<Type, decltype(f), 2>;
  const typename Evaluator::Inputs inputs = {std::cref(TTa), std::cref(TTb)};

  // index sets of y and (different) index sets of the error approximation
  std::mt19937 gen(17);
  const auto I = randomLeftIndexSets(dims, {2, 3, 2, 3}, gen);
  const auto J = randomRightIndexSets(dims, {0, 3, 2, 3, 2}, gen);
  const auto Iz = randomLeftIndexSets(dims, {1, 2, 2, 2}, gen);
  const auto Jz = randomRightIndexSets(dims, {0, 2, 3, 2, 1}, gen);

  // compare a block with f evaluated directly on the cross (L, modes of the cores k..k+nModes-1, R)
  long long nEntries = 0;
  const auto checkBlock = [&](const PITTS::Tensor3<Type>& B, const Eigen::MatrixXi& L, int k, int nModes, const Eigen::MatrixXi& R)
  {
    ASSERT_EQ(L.rows(), B.r1());
    ASSERT_EQ(R.rows(), B.r2());
    ASSERT_EQ(nModes == 1 ? dims[k] : dims[k]*dims[k+1], B.n());
    nEntries += B.r1()*B.n()*B.r2();
    for(int a = 0; a < B.r1(); a++)
      for(int b = 0; b < B.r2(); b++)
        for(int j = 0; j < B.n(); j++)
        {
          Eigen::RowVectorXi multiIndex(nDim);
          if( nModes == 2 )
            multiIndex << L.row(a), j % dims[k], j / dims[k], R.row(b);
          else
            multiIndex << L.row(a), j, R.row(b);
          const Type ref = f(evaluate(TTa, multiIndex), evaluate(TTb, multiIndex));
          EXPECT_NEAR(ref, B(a, j, b), eps*(1 + std::abs(ref)));
        }
  };

  for(int nMALS: {1, 2})
    for(int k = 0; k + nMALS <= nDim; k++)
    {
      // left-to-right step: block, then the enrichment crosses (as in the driver)
      Evaluator fromLeft(f, inputs, I, J, Iz, Jz);
      nEntries = 0;
      checkBlock(fromLeft.blockFromLeft(k, nMALS), expandLeft(I, k-1), k, nMALS, expandRight(J, k+nMALS, nDim));
      if( nMALS == 1 )
      {
        checkBlock(fromLeft.mixedBlock(k), expandLeft(I, k-1), k, 1, expandRight(Jz, k+1, nDim));
        checkBlock(fromLeft.errorBlockFromLeft(k), expandLeft(Iz, k-1), k, 1, expandRight(Jz, k+1, nDim));
      }
      EXPECT_EQ(nEntries, fromLeft.nEvaluations());

      // right-to-left step: the same crosses with the contractions from the other side
      Evaluator fromRight(f, inputs, I, J, Iz, Jz);
      checkBlock(fromRight.blockFromRight(k, nMALS), expandLeft(I, k-1), k, nMALS, expandRight(J, k+nMALS, nDim));
      if( nMALS == 1 )
        checkBlock(fromRight.errorBlockFromRight(k), expandLeft(Iz, k-1), k, 1, expandRight(Jz, k+1, nDim));
    }

  // random entries (control sample)
  Evaluator samples(f, inputs, I, J, Iz, Jz);
  const Eigen::MatrixXi indices = randomIndices(dims, 10);
  const PITTS::Tensor3<Type> values = samples.atIndices(indices);
  ASSERT_EQ(10, values.r1());
  for(int s = 0; s < 10; s++)
  {
    const Type ref = f(evaluate(TTa, indices.row(s)), evaluate(TTb, indices.row(s)));
    EXPECT_NEAR(ref, values(s, 0, 0), eps*(1 + std::abs(ref)));
  }
  EXPECT_EQ(10, samples.nEvaluations());
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

  auto Lx = defineSweepData<PITTS::Tensor3<Type>>(nDim, contraction_loop_from_left<Type>(TT, I), contraction_loop_from_right<Type>(TT, J));
  Lx.update(2, 3);

  // change I_1 (which makes I_2 point to new parents): LX_2 = L_1 X_2 depends on it, so invalidate(2) + update recomputes it
  I[1] = randomLeftIndexSets(dims, {2, 3}, gen)[1];
  Lx.invalidate(2);
  Lx.update(2, 3);

  const Eigen::MatrixXi I2 = expandLeft(I, 2);
  const PITTS::Tensor2<Type> L2 = interfaceLeft(Lx.subTensor(2), I[2]);
  ASSERT_EQ(2, L2.r1());
  // check the recomputed L_2 through the product with R_3
  // (an Eigen product is lazy, so store it in a matrix and not in an auto variable)
  const Eigen::MatrixXi J3 = expandRight(J, 3, nDim);
  const PITTS::Tensor2<Type> R3 = interfaceRight(Lx.subTensor(3), J[3]);
  const Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> LR = ConstEigenMap(L2) * ConstEigenMap(R3);
  for(int a = 0; a < I2.rows(); a++)
    for(int b = 0; b < J3.rows(); b++)
    {
      Eigen::RowVectorXi multiIndex(nDim);
      multiIndex << I2.row(a), J3.row(b);
      EXPECT_NEAR(evaluate(TT, multiIndex), LR(a, b), eps);
    }
}


// ---- block helpers ----

using PITTS::internal::tt_cross::applyToBlock;
using PITTS::internal::tt_cross::t3_rms;
using PITTS::internal::tt_cross::rmsDifference;
using PITTS::internal::tt_cross::randomIndices;
using PITTS::internal::tt_cross::evaluateAt;

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, randomIndices_and_evaluateAt)
{
  using Type = typename TestFixture::Type;
  const auto eps = 1.e-6;

  const std::vector<int> dims = {3, 4, 2, 5, 3};
  PITTS::TensorTrain<Type> TT(dims);
  TT.setTTranks({2, 3, 2, 4});
  randomize(TT);

  const int nSamples = 20;
  const Eigen::MatrixXi indices = randomIndices(dims, nSamples);
  ASSERT_EQ(nSamples, indices.rows());
  ASSERT_EQ(dims.size(), indices.cols());
  for(int k = 0; k < int(dims.size()); k++)
  {
    EXPECT_GE(indices.col(k).minCoeff(), 0);
    EXPECT_LT(indices.col(k).maxCoeff(), dims[k]);
  }

  // same as the scalar products with the corresponding unit tensors
  PITTS::Tensor3<Type> values;
  evaluateAt(TT, indices, values);
  ASSERT_EQ(nSamples, values.r1());
  ASSERT_EQ(1, values.n());
  ASSERT_EQ(1, values.r2());
  for(int s = 0; s < nSamples; s++)
  {
    PITTS::TensorTrain<Type> unit(dims);
    unit.setUnit(std::vector<int>(indices.row(s).begin(), indices.row(s).end()));
    EXPECT_NEAR(dot(unit, TT), values(s, 0, 0), eps);
  }
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, applyToBlock)
{
  using Type = typename TestFixture::Type;
  constexpr auto eps = 1.e-6;

  PITTS::Tensor3<Type> B(3, 4, 2);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 2; k++)
        B(i, j, k) = Type(1 + i + 3*j + 12*k);

  const long long nEval = applyToBlock([](Type v){return v*v + Type(1);}, B);
  EXPECT_EQ(3*4*2, nEval);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 2; k++)
      {
        const Type v = Type(1 + i + 3*j + 12*k);
        EXPECT_NEAR(v*v + Type(1), B(i, j, k), eps);
      }

  // larger block (parallel path) with a non-trivial function
  PITTS::Tensor3<Type> C(5, 40, 7);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 40; j++)
      for(int k = 0; k < 7; k++)
        C(i, j, k) = Type(1 + i + j + k);
  EXPECT_EQ(5*40*7, applyToBlock([](Type v){return std::sqrt(v);}, C));
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 40; j++)
      for(int k = 0; k < 7; k++)
        EXPECT_NEAR(std::sqrt(Type(1 + i + j + k)), C(i, j, k), eps);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, applyToBlock_multivector)
{
  using Type = typename TestFixture::Type;
  constexpr auto eps = 1.e-6;

  // the array variant sees all entries of the block as one column, ordered as in the Tensor3 memory
  PITTS::Tensor3<Type> B(3, 4, 2);
  const long long nEval = applyToBlock([](PITTS::MultiVector<Type>& mv)
  {
    EXPECT_EQ(3*4*2, mv.rows());
    EXPECT_EQ(1, mv.cols());
    for(long long i = 0; i < mv.rows(); i++)
      mv(i, 0) = Type(i);
  }, B);
  EXPECT_EQ(3*4*2, nEval);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 2; k++)
        EXPECT_NEAR(Type(i + 3*j + 12*k), B(i, j, k), eps);

  // same result as the scalar variant
  PITTS::Tensor3<Type> C(5, 40, 7), Cref(5, 40, 7);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 40; j++)
      for(int k = 0; k < 7; k++)
        C(i, j, k) = Cref(i, j, k) = Type(1 + i + j + k);

  EXPECT_EQ(5*40*7, applyToBlock([](PITTS::MultiVector<Type>& mv)
  {
    for(long long i = 0; i < mv.rows(); i++)
      mv(i, 0) = std::sqrt(mv(i, 0));
  }, C));
  applyToBlock([](Type v){return std::sqrt(v);}, Cref);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 40; j++)
      for(int k = 0; k < 7; k++)
        EXPECT_NEAR(Cref(i, j, k), C(i, j, k), eps);

  // the array function must not change the dimensions of the block
  PITTS::Tensor3<Type> D(2, 3, 2);
  EXPECT_THROW(applyToBlock([](PITTS::MultiVector<Type>& mv){mv.resize(7, 1);}, D), std::invalid_argument);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, applyToBlock_two_arguments)
{
  using Type = typename TestFixture::Type;
  constexpr auto eps = 1.e-6;

  // entries a(i,j,k) and b(i,j,k) of two blocks, small (serial) and large (parallel path)
  for(const auto [r1, n, r2]: {std::array<int,3>{3, 4, 2}, std::array<int,3>{5, 40, 7}})
  {
    const auto a = [](int i, int j, int k){return Type(1 + i + 3*j + 12*k);};
    const auto b = [](int i, int j, int k){return Type(2 + 5*i + j + k);};
    PITTS::Tensor3<Type> A(r1, n, r2), Aarray(r1, n, r2), B(r1, n, r2), Barray(r1, n, r2);
    for(int i = 0; i < r1; i++)
      for(int j = 0; j < n; j++)
        for(int k = 0; k < r2; k++)
        {
          A(i, j, k) = Aarray(i, j, k) = a(i, j, k);
          B(i, j, k) = Barray(i, j, k) = b(i, j, k);
        }

    // scalar function f(a, b): the arguments in the order of the blocks
    EXPECT_EQ(r1*n*r2, applyToBlock([](Type x, Type y){return x - Type(2)*y;}, A, std::move(B)));

    // array function f(a, b): the second argument sees the entries of the second block in the Tensor3 memory order
    EXPECT_EQ(r1*n*r2, applyToBlock([](PITTS::MultiVector<Type>& x, const PITTS::MultiVector<Type>& y)
    {
      ASSERT_EQ(x.rows(), y.rows());
      ASSERT_EQ(1, y.cols());
      for(long long i = 0; i < x.rows(); i++)
        x(i, 0) = x(i, 0) - Type(2)*y(i, 0);
    }, Aarray, std::move(Barray)));

    for(int i = 0; i < r1; i++)
      for(int j = 0; j < n; j++)
        for(int k = 0; k < r2; k++)
        {
          EXPECT_NEAR(a(i, j, k) - Type(2)*b(i, j, k), A(i, j, k), eps);
          EXPECT_NEAR(a(i, j, k) - Type(2)*b(i, j, k), Aarray(i, j, k), eps);
        }
  }

  // the blocks must have the same shape
  PITTS::Tensor3<Type> C(2, 3, 2), D(2, 3, 3);
  EXPECT_THROW(applyToBlock([](Type x, Type y){return x*y;}, C, std::move(D)), std::invalid_argument);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, t3_rms_and_rmsDifference)
{
  using Type = typename TestFixture::Type;
  constexpr auto eps = 1.e-6;

  // constant blocks: the rms is the entry value, independent of the shape (unlike the Frobenius norm)
  for(const auto& shape: std::vector<std::array<int,3>>{{2, 3, 2}, {7, 3, 5}, {1, 1, 1}})
  {
    PITTS::Tensor3<Type> constant(shape[0], shape[1], shape[2]);
    constant.setConstant(Type(2.5));
    EXPECT_NEAR(2.5, t3_rms(constant), eps);
  }

  PITTS::Tensor3<Type> A(2, 3, 2), B(2, 3, 2), C(2, 3, 2);
  Type sumSquares = 0;
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
      {
        B(i, j, k) = Type(1 + i + 2*j + 6*k);
        A(i, j, k) = B(i, j, k);
        C(i, j, k) = Type(1.5) * B(i, j, k);
        sumSquares += B(i, j, k)*B(i, j, k);
      }

  EXPECT_NEAR(std::sqrt(sumSquares/12), t3_rms(B), eps);
  EXPECT_NEAR(0, rmsDifference(A, B), eps);
  // rms(1.5 B - B) = 0.5 rms(B)
  EXPECT_NEAR(0.5*t3_rms(B), rmsDifference(C, B), eps);

  PITTS::Tensor3<Type> wrongShape(2, 3, 3);
  EXPECT_THROW(rmsDifference(wrongShape, B), std::invalid_argument);
}


// ---- local core updates (y in the orthogonal gauge) ----

#include "pitts_tensor3_fold.hpp"
#include <limits>
#include <optional>

using PITTS::internal::tt_cross::updateCoreLeft;
using PITTS::internal::tt_cross::updateCoreRight;
using PITTS::internal::tt_cross::CoreUpdate;

namespace
{
  template<typename T>
  PITTS::Tensor3<T> randomBlock(int r1, int n, int r2)
  {
    PITTS::Tensor3<T> B(r1, n, r2);
    EigenMap(unfold_left(B)).setRandom();
    return B;
  }

  template<typename T>
  auto blockTolerance()
  {
    return 10*std::sqrt(std::numeric_limits<T>::epsilon());
  }

  // interface matrix of the boundary: the identity (here also for blocks with more than one row/column)
  template<typename T>
  PITTS::Tensor2<T> eye(int n)
  {
    PITTS::Tensor2<T> I(n, n);
    EigenMap(I) = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
    return I;
  }

  // distinct indices in [0, n)
  void checkSelection(const std::vector<int>& selected, int nExpected, long long n)
  {
    ASSERT_EQ(nExpected, selected.size());
    const std::set<int> unique(selected.begin(), selected.end());
    EXPECT_EQ(nExpected, unique.size());
    for(int i: selected)
    {
      EXPECT_LE(0, i);
      EXPECT_LT(i, n);
    }
  }

  // common invariants of a left update: left-orthonormal core, core*carry reconstructs C, q selected rows
  template<typename T>
  void checkLeftUpdate(const PITTS::Tensor3<T>& C, const CoreUpdate<T>& upd, bool exactReconstruction, int nRows = 0)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    const auto eps = blockTolerance<T>();
    const int r = upd.core.r2();
    ASSERT_EQ(C.r1(), upd.core.r1());
    ASSERT_EQ(C.n(), upd.core.n());
    ASSERT_EQ(r, upd.carry.r1());
    ASSERT_EQ(C.r2(), upd.carry.r2());
    checkSelection(upd.selected, nRows > 0 ? nRows : r, C.r1()*C.n());

    const auto mapCore = ConstEigenMap(unfold_left(upd.core));
    EXPECT_NEAR(mat::Identity(r, r), mat(mapCore.adjoint() * mapCore), eps);
    if( exactReconstruction )
    {
      EXPECT_NEAR(ConstEigenMap(unfold_left(C)), mat(mapCore * ConstEigenMap(upd.carry)), eps);
    }
  }

  // common invariants of a right update: right-orthonormal core, carry*core reconstructs C, q selected columns
  template<typename T>
  void checkRightUpdate(const PITTS::Tensor3<T>& C, const CoreUpdate<T>& upd, bool exactReconstruction, int nColumns = 0)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    const auto eps = blockTolerance<T>();
    const int r = upd.core.r1();
    ASSERT_EQ(C.n(), upd.core.n());
    ASSERT_EQ(C.r2(), upd.core.r2());
    ASSERT_EQ(C.r1(), upd.carry.r1());
    ASSERT_EQ(r, upd.carry.r2());
    checkSelection(upd.selected, nColumns > 0 ? nColumns : r, C.n()*C.r2());

    const auto mapCore = ConstEigenMap(unfold_right(upd.core));
    EXPECT_NEAR(mat::Identity(r, r), mat(mapCore * mapCore.adjoint()), eps);
    if( exactReconstruction )
    {
      EXPECT_NEAR(ConstEigenMap(unfold_right(C)), mat(ConstEigenMap(upd.carry) * mapCore), eps);
    }
  }
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_full_rank)
{
  using Type = typename TestFixture::Type;
  const Type maxvolTol = Type(0.05);

  const auto C = randomBlock<Type>(2, 4, 3);
  const CoreUpdate<Type> upd = updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(0), std::numeric_limits<int>::max(), maxvolTol);
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(3, upd.core.r2());
  checkLeftUpdate(C, upd, true);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_low_rank_and_maxRank)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);

  // exact rank 2 block (r1*n = 8, r2 = 3)
  PITTS::Tensor3<Type> C(2, 4, 3);
  EigenMap(unfold_left(C)) = mat::Random(8, 2) * mat::Random(2, 3);

  const CoreUpdate<Type> upd = updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol);
  EXPECT_EQ(2, upd.truncatedRank);
  EXPECT_EQ(2, upd.core.r2());
  checkLeftUpdate(C, upd, true);

  // maxRank limits the rank, no exact reconstruction then
  const CoreUpdate<Type> upd1 = updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(0), 1, maxvolTol);
  EXPECT_EQ(1, upd1.truncatedRank);
  EXPECT_EQ(1, upd1.core.r2());
  checkLeftUpdate(C, upd1, false);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_wide_block)
{
  using Type = typename TestFixture::Type;
  const Type maxvolTol = Type(0.05);

  // r1*n = 2 < r2 = 5: the rank is limited by the number of rows
  const auto C = randomBlock<Type>(1, 2, 5);
  const CoreUpdate<Type> upd = updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(0), std::numeric_limits<int>::max(), maxvolTol);
  EXPECT_EQ(2, upd.core.r2());
  checkLeftUpdate(C, upd, true);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_enrichment)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);
  const auto eps = blockTolerance<Type>();

  const auto C = randomBlock<Type>(2, 4, 3);
  const auto E = randomBlock<Type>(2, 4, 2);   // the same block shape, 2 candidate directions
  PITTS::Tensor3<Type> Econsumed;              // updateCoreLeft consumes the directions
  copy(E, Econsumed);

  const CoreUpdate<Type> upd = updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(0), std::numeric_limits<int>::max(), maxvolTol, std::move(Econsumed));
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(5, upd.core.r2());
  // the enrichment does not change the represented block (zero rows in the carry)
  checkLeftUpdate(C, upd, true);

  // the columns of the core span the enrichment directions: E - core core^H E = 0
  const auto mapCore = ConstEigenMap(unfold_left(upd.core));
  const auto mapE = ConstEigenMap(unfold_left(E));
  EXPECT_NEAR(mat::Zero(8, 2), mat(mapE - mapCore * (mapCore.adjoint() * mapE)), eps);

  // limited by maxEnrichment and by maxEnrichedRank
  EXPECT_EQ(4, updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(0), std::numeric_limits<int>::max(), maxvolTol, randomBlock<Type>(2, 4, 2), Type(0), 1).core.r2());
  EXPECT_EQ(4, updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(0), std::numeric_limits<int>::max(), maxvolTol, randomBlock<Type>(2, 4, 2), Type(0), 2, 4).core.r2());
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_enrichment_already_in_span)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);

  const auto C = randomBlock<Type>(2, 4, 3);
  // directions inside the column space of C: nothing to add
  PITTS::Tensor3<Type> E(2, 4, 2);
  EigenMap(unfold_left(E)) = ConstEigenMap(unfold_left(C)) * mat::Random(3, 2);

  const CoreUpdate<Type> upd = updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(0), std::numeric_limits<int>::max(), maxvolTol, std::move(E));
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(3, upd.core.r2());
  checkLeftUpdate(C, upd, true);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_enrichment_limited_by_rows)
{
  using Type = typename TestFixture::Type;
  const Type maxvolTol = Type(0.05);

  // the left unfolding has only r1*n = 4 rows: at most 4 - rank directions can be added
  const auto C = randomBlock<Type>(2, 2, 3);
  const CoreUpdate<Type> upd = updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(0), std::numeric_limits<int>::max(), maxvolTol, randomBlock<Type>(2, 2, 2));
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(4, upd.core.r2());
  checkLeftUpdate(C, upd, true);

  // full rank already: nothing can be added (the projected remainder is only round-off)
  const auto Cfull = randomBlock<Type>(1, 3, 5);
  const CoreUpdate<Type> updFull = updateCoreLeft<Type>(Cfull, eye<Type>(Cfull.r1()), Type(0), std::numeric_limits<int>::max(), maxvolTol, randomBlock<Type>(1, 3, 2));
  EXPECT_EQ(3, updFull.truncatedRank);
  EXPECT_EQ(3, updFull.core.r2());
  checkLeftUpdate(Cfull, updFull, true);
}


TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, enrichFactors_inexactly_orthogonal_factor)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using PITTS::internal::tt_cross::enrichFactors;
  const Type machEps = std::numeric_limits<Type>::epsilon();

  // internal::normalize_svd does not compute the orthogonal factor of a block but reconstructs it as
  // Q = M V S^{-1}, so ||Q^H Q - I|| is only of the order eps*sigma_1/sigma_r. Mimic that here with a
  // prescribed orthogonality error, well above the sqrt(eps) below which the enrichment drops a direction.
  const int r1 = 3, n = 4, r = 4;
  const Type orthogonalityError = std::pow(machEps, Type(0.25));
  mat Qmat = mat(mat::Random(r1*n, r).householderQr().householderQ()).leftCols(r);
  Qmat.col(r-1) += orthogonalityError * Qmat.col(0);

  PITTS::Tensor3<Type> Q(r1, n, r);
  EigenMap(unfold_left(Q)) = Qmat;
  PITTS::Tensor2<Type> B(r, 2);
  EigenMap(B) = mat::Random(r, 2);

  // candidate directions that lie exactly in the span of Q: nothing may be added. Projecting them out as
  // E - Q (Q^H E) leaves the orthogonality error instead of zero, which is far above the drop tolerance,
  // so directions of pure round-off get appended and the enriched factor becomes rank-deficient.
  PITTS::Tensor3<Type> E(r1, n, 2);
  EigenMap(unfold_left(E)) = Qmat * mat::Random(r, 2);

  enrichFactors(Q, B, std::move(E));

  EXPECT_EQ(r, Q.r2());
  EXPECT_EQ(r, B.r1());
  EXPECT_NEAR(Qmat, ConstEigenMap(unfold_left(Q)), Type(0));
}


TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreRight_full_rank)
{
  using Type = typename TestFixture::Type;
  const Type maxvolTol = Type(0.05);

  const auto C = randomBlock<Type>(3, 4, 2);
  const CoreUpdate<Type> upd = updateCoreRight<Type>(C, eye<Type>(C.r2()), Type(0), std::numeric_limits<int>::max(), maxvolTol);
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(3, upd.core.r1());
  checkRightUpdate(C, upd, true);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreRight_low_rank_and_maxRank)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);

  PITTS::Tensor3<Type> C(3, 4, 2);
  EigenMap(unfold_right(C)) = mat::Random(3, 2) * mat::Random(2, 8);

  const CoreUpdate<Type> upd = updateCoreRight<Type>(C, eye<Type>(C.r2()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol);
  EXPECT_EQ(2, upd.truncatedRank);
  EXPECT_EQ(2, upd.core.r1());
  checkRightUpdate(C, upd, true);

  const CoreUpdate<Type> upd1 = updateCoreRight<Type>(C, eye<Type>(C.r2()), Type(0), 1, maxvolTol);
  EXPECT_EQ(1, upd1.core.r1());
  checkRightUpdate(C, upd1, false);
}


// ---- oversampled selection and interface matrices ----

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_oversampling)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);
  const auto eps = blockTolerance<Type>();

  // exact rank 3 block (r1*n = 24, r2 = 6): oversampling 1 selects 3 + 3 rows
  PITTS::Tensor3<Type> C(3, 8, 6);
  EigenMap(unfold_left(C)) = mat::Random(24, 3) * mat::Random(3, 6);
  const CoreUpdate<Type> square = updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol);
  const CoreUpdate<Type> upd = updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol, PITTS::Tensor3<Type>(), Type(1));
  EXPECT_EQ(3, upd.truncatedRank);
  checkLeftUpdate(C, upd, true, 6);
  // oversampling only changes the selection: the same core, and the first rows are the square maxvol rows
  EXPECT_NEAR(ConstEigenMap(unfold_left(square.core)), ConstEigenMap(unfold_left(upd.core)), eps);
  EXPECT_EQ(square.selected, std::vector<int>(upd.selected.begin(), upd.selected.begin() + 3));

  // fractional oversampling is rounded up: 3 + ceil(0.1 * 3) = 4 rows
  checkLeftUpdate(C, updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol, PITTS::Tensor3<Type>(), Type(0.1)), true, 4);
  // limited by the number of rows of the block and by maxRows (but at least the rank)
  checkLeftUpdate(C, updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol, PITTS::Tensor3<Type>(), Type(100)), true, 24);
  checkLeftUpdate(C, updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol, PITTS::Tensor3<Type>(), Type(1), 0, 100, 5), true, 5);
  checkLeftUpdate(C, updateCoreLeft<Type>(C, eye<Type>(C.r1()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol, PITTS::Tensor3<Type>(), Type(1), 0, 100, 1), true, 3);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreRight_oversampling)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);

  // exact rank 3 block (r1 = 6, n*r2 = 24)
  PITTS::Tensor3<Type> C(6, 8, 3);
  EigenMap(unfold_right(C)) = mat::Random(6, 3) * mat::Random(3, 24);
  const CoreUpdate<Type> square = updateCoreRight<Type>(C, eye<Type>(C.r2()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol);
  const CoreUpdate<Type> upd = updateCoreRight<Type>(C, eye<Type>(C.r2()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol, Type(1));
  EXPECT_EQ(3, upd.truncatedRank);
  checkRightUpdate(C, upd, true, 6);
  EXPECT_EQ(square.selected, std::vector<int>(upd.selected.begin(), upd.selected.begin() + 3));

  checkRightUpdate(C, updateCoreRight<Type>(C, eye<Type>(C.r2()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol, Type(0.1)), true, 4);
  checkRightUpdate(C, updateCoreRight<Type>(C, eye<Type>(C.r2()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol, Type(100)), true, 24);
  checkRightUpdate(C, updateCoreRight<Type>(C, eye<Type>(C.r2()), Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol, Type(1), 4), true, 4);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCore_selection_with_interface)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);
  const auto eps = blockTolerance<Type>();

  // left: the rows are selected from M = (L x I_n) U, the values of the new left part on I_{k-1} x S_k;
  // without oversampling M(rows,:) is a dominant submatrix
  const int q = 5, r1 = 3, n = 4, r2 = 4;
  PITTS::Tensor2<Type> L(q, r1);
  EigenMap(L) = mat::Random(q, r1);
  const auto C = randomBlock<Type>(r1, n, r2);
  const CoreUpdate<Type> upd = updateCoreLeft<Type>(C, L, Type(0), std::numeric_limits<int>::max(), maxvolTol);
  checkSelection(upd.selected, r2, q*n);
  {
    PITTS::Tensor3<Type> M;
    PITTS::internal::normalize_contract1(L, upd.core, M);
    const auto mapM = ConstEigenMap(unfold_left(M));
    mat Mhat(r2, r2);
    for(int j = 0; j < r2; j++)
      Mhat.row(j) = mapM.row(upd.selected[j]);
    EXPECT_LE(mat(mapM * Mhat.inverse()).array().abs().maxCoeff(), 1 + maxvolTol + eps);
  }
  // oversampled: more rows of M
  checkSelection(updateCoreLeft<Type>(C, L, Type(0), std::numeric_limits<int>::max(), maxvolTol, PITTS::Tensor3<Type>(), Type(1)).selected, 2*r2, q*n);

  // right: the columns are selected from M = Vt (I_n x R)
  PITTS::Tensor2<Type> R(r2, q);
  EigenMap(R) = mat::Random(r2, q);
  const auto C2 = randomBlock<Type>(r1, n, r2);
  const CoreUpdate<Type> updR = updateCoreRight<Type>(C2, R, Type(0), std::numeric_limits<int>::max(), maxvolTol);
  checkSelection(updR.selected, r1, n*q);
  {
    PITTS::Tensor3<Type> M;
    PITTS::internal::normalize_contract2(updR.core, R, M);
    const auto mapM = ConstEigenMap(unfold_right(M));
    mat Mhat(r1, r1);
    for(int j = 0; j < r1; j++)
      Mhat.col(j) = mapM.col(updR.selected[j]);
    EXPECT_LE(mat(Mhat.inverse() * mapM).array().abs().maxCoeff(), 1 + maxvolTol + eps);
  }

  // wrong interface dimensions
  EXPECT_THROW(updateCoreLeft<Type>(randomBlock<Type>(2, n, r2), L, Type(0), 10, maxvolTol), std::invalid_argument);
  EXPECT_THROW(updateCoreRight<Type>(randomBlock<Type>(r1, n, 2), R, Type(0), 10, maxvolTol), std::invalid_argument);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, solveLeftInterface_solveRightInterface)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using PITTS::internal::tt_cross::solveLeftInterface;
  using PITTS::internal::tt_cross::solveRightInterface;
  const auto eps = blockTolerance<Type>();

  // inverse operations of normalize_contract1 / normalize_contract2 with a full-rank interface matrix: square and
  // oversampled (least squares, exact for a block in the range of the interface); the boundary is the 1x1 identity
  const auto X = randomBlock<Type>(3, 2, 3);
  for(int q: {3, 5})
  {
    PITTS::Tensor2<Type> L(q, 3), R(3, q);
    EigenMap(L) = mat::Random(q, 3);
    EigenMap(R) = mat::Random(3, q);
    PITTS::Tensor3<Type> B, Xsolved;
    PITTS::internal::normalize_contract1(L, X, B);
    solveLeftInterface(L, B, Xsolved);
    EXPECT_NEAR(ConstEigenMap(unfold_left(X)), ConstEigenMap(unfold_left(Xsolved)), eps);
    PITTS::internal::normalize_contract2(X, R, B);
    solveRightInterface(B, R, Xsolved);
    EXPECT_NEAR(ConstEigenMap(unfold_left(X)), ConstEigenMap(unfold_left(Xsolved)), eps);

    // least squares: a perturbation outside the range of L (resp. R^H) does not change the solution
    PITTS::Tensor3<Type> Bl;
    PITTS::internal::normalize_contract1(L, X, Bl);
    const mat Pperp = mat::Identity(q, q) - ConstEigenMap(L) * ConstEigenMap(L).completeOrthogonalDecomposition().pseudoInverse();
    EigenMap(unfold_right(Bl)) += Pperp * mat::Random(q, 6);
    solveLeftInterface(L, Bl, Xsolved);
    EXPECT_NEAR(ConstEigenMap(unfold_left(X)), ConstEigenMap(unfold_left(Xsolved)), eps);
  }
  // boundary
  PITTS::Tensor3<Type> copyLeft, copyRight;
  const auto X1 = randomBlock<Type>(1, 2, 1);
  solveLeftInterface(eye<Type>(1), X1, copyLeft);
  solveRightInterface(X1, eye<Type>(1), copyRight);
  EXPECT_NEAR(ConstEigenMap(unfold_left(X1)), ConstEigenMap(unfold_left(copyLeft)), eps);
  EXPECT_NEAR(ConstEigenMap(unfold_left(X1)), ConstEigenMap(unfold_left(copyRight)), eps);
  // dimension mismatch
  EXPECT_THROW(solveLeftInterface(eye<Type>(2), X, copyLeft), std::invalid_argument);
  EXPECT_THROW(solveRightInterface(X, eye<Type>(2), copyRight), std::invalid_argument);
}


TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, contractInterfaces_solveInterfaces)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using PITTS::internal::tt_cross::boundaryInterface;
  using PITTS::internal::tt_cross::contractInterfaces;
  using PITTS::internal::tt_cross::solveInterfaces;
  const auto eps = blockTolerance<Type>();

  // L X R on both sides at once, and its inverse (q = 0: boundary; square and oversampled interfaces)
  for(int q: {0, 3, 5})
  {
    PITTS::Tensor2<Type> L = boundaryInterface<Type>(), R = boundaryInterface<Type>();
    const auto Xq = q == 0 ? randomBlock<Type>(1, 2, 1) : randomBlock<Type>(3, 2, 4);
    if( q > 0 )
    {
      L.resize(q, 3);
      R.resize(4, q+1);
      EigenMap(L) = mat::Random(q, 3);
      EigenMap(R) = mat::Random(4, q+1);
    }
    PITTS::Tensor3<Type> B, Bref, withLeft, Xsolved;
    contractInterfaces(L, Xq, R, B);
    PITTS::internal::normalize_contract1(L, Xq, withLeft);
    PITTS::internal::normalize_contract2(withLeft, R, Bref);
    EXPECT_NEAR(ConstEigenMap(unfold_left(Bref)), ConstEigenMap(unfold_left(B)), eps);
    solveInterfaces(L, B, R, Xsolved);
    EXPECT_NEAR(ConstEigenMap(unfold_left(Xq)), ConstEigenMap(unfold_left(Xsolved)), eps);
  }
  EXPECT_EQ(1, boundaryInterface<Type>().r1());
  EXPECT_EQ(1, boundaryInterface<Type>().r2());
  EXPECT_EQ(Type(1), boundaryInterface<Type>()(0, 0));
}


// ---- index selection without cores (error approximation) ----

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, dominantRows_dominantColumns)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using PITTS::internal::tt_cross::dominantRows;
  using PITTS::internal::tt_cross::dominantColumns;
  const Type maxvolTol = Type(0.05);

  // exact rank 2 block: always the requested number of rows/columns; the first two are the square maxvol selection
  // of the 2 dominant directions, the others are added by oversampling
  PITTS::Tensor3<Type> B(3, 4, 5);
  EigenMap(unfold_left(B)) = mat::Random(12, 2) * mat::Random(2, 5);
  const std::vector<int> rows = dominantRows(B, Type(1.e-4), 4, maxvolTol);
  checkSelection(rows, 4, 12);
  const std::vector<int> rows2 = dominantRows(B, Type(1.e-4), 2, maxvolTol);
  checkSelection(rows2, 2, 12);
  EXPECT_EQ(rows2, std::vector<int>(rows.begin(), rows.begin()+2));
  checkSelection(dominantRows(B, Type(0), 1, maxvolTol), 1, 12);
  PITTS::Tensor3<Type> B2(3, 4, 5);
  EigenMap(unfold_right(B2)) = mat::Random(3, 2) * mat::Random(2, 20);
  const std::vector<int> columns = dominantColumns(B2, Type(1.e-4), 3, maxvolTol);
  checkSelection(columns, 3, 20);
  const std::vector<int> columns2 = dominantColumns(B2, Type(1.e-4), 2, maxvolTol);
  checkSelection(columns2, 2, 20);
  EXPECT_EQ(columns2, std::vector<int>(columns.begin(), columns.begin()+2));
  checkSelection(dominantColumns(B2, Type(0), 1, maxvolTol), 1, 20);

  // directions below the tolerance (here: round-off of a rank-deficient residual) do not change the selection
  PITTS::Tensor3<Type> Bnoise(3, 4, 5), B2noise(3, 4, 5);
  EigenMap(unfold_left(Bnoise)) = ConstEigenMap(unfold_left(B)) + 10*std::numeric_limits<Type>::epsilon() * mat::Random(12, 5);
  EigenMap(unfold_right(B2noise)) = ConstEigenMap(unfold_right(B2)) + 10*std::numeric_limits<Type>::epsilon() * mat::Random(3, 20);
  EXPECT_EQ(rows, dominantRows(Bnoise, Type(1.e-4), 4, maxvolTol));
  EXPECT_EQ(columns, dominantColumns(B2noise, Type(1.e-4), 3, maxvolTol));
}


// ---- initial right frame of y and its index sets ----

#include "pitts_tensortrain_normalize.hpp"

using PITTS::internal::tt_cross::initialRightFrame;
using PITTS::internal::tt_cross::rightIndexSetsFromFrame;

namespace
{
  // max |A^H (A(:,cols))^-H| for the right unfolding A = unfold_right(X_k * R_{k+1}): the dominance of the selected columns
  template<typename T>
  auto columnDominance(const PITTS::Tensor3<T>& XR, const NestedIndices& Jk)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    const auto mapA = ConstEigenMap(unfold_right(XR));
    const int r = Jk.size();
    mat Ahat(mapA.rows(), r);
    for(int b = 0; b < r; b++)
      Ahat.col(b) = mapA.col(Jk.idx[b] + Jk.link[b]*XR.n());
    const mat B = Ahat.inverse() * mapA;
    return B.array().abs().maxCoeff();
  }

  // smallest singular value of the right interface R_k (r_{k-1} x |J_k|) of a TT relative to the largest one, for k = 1..d-1:
  // a positive value means that the cross determines the right part of the TT
  template<typename T>
  std::vector<T> relativeSmallestSingularValuesOfRightInterfaces(const PITTS::TensorTrain<T>& TT, const std::vector<NestedIndices>& J)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    const int nDim = TT.dimensions().size();
    std::vector<NestedIndices> noLeft(nDim);
    auto Rx = defineSweepData<PITTS::Tensor3<T>>(nDim, contraction_loop_from_left<T>(TT, noLeft), contraction_loop_from_right<T>(TT, J));
    Rx.update(-1, 1);
    std::vector<T> result(nDim, T(0));
    for(int k = 1; k < nDim; k++)
    {
      const PITTS::Tensor2<T> R = interfaceRight(Rx.subTensor(k), J[k]);
      const Eigen::JacobiSVD<mat> svd(mat(ConstEigenMap(R)));
      const auto& sigma = svd.singularValues();
      result[k] = sigma(sigma.size()-1) / sigma(0);
    }
    return result;
  }

  // frame of the inputs and its index sets
  template<typename T>
  std::vector<NestedIndices> indexSetsOfFrame(const std::vector<std::reference_wrapper<const PITTS::TensorTrain<T>>>& inputs, T maxvolTol,
                                              int maxRank = std::numeric_limits<int>::max(), T oversampling = T(0), const std::vector<int>& maxIndices = {})
  {
    PITTS::TensorTrain<T> frame(inputs.front().get().dimensions());
    initialRightFrame(inputs, frame, maxRank);
    return rightIndexSetsFromFrame(frame, maxvolTol, oversampling, maxIndices);
  }
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, initialRightFrame)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using Inputs = std::vector<std::reference_wrapper<const PITTS::TensorTrain<Type>>>;
  const auto eps = blockTolerance<Type>();

  const std::vector<int> dims = {3, 4, 3, 5};
  PITTS::TensorTrain<Type> TTa(dims), TTb(dims);
  TTa.setTTranks({2, 3, 2});
  TTb.setTTranks({2, 2, 3});
  PITTS::internal::randomGenerator.seed(4711);   // same data on all MPI processes (PITTS::initialize seeds them differently)
  randomize(TTa);
  randomize(TTb);

  // y = 0 with right-orthonormal cores 1..d-1, of the ranks of the inputs (summed, at most the full ranks or maxRank)
  const auto check = [&](const Inputs& inputs, const std::vector<int>& expectedRanks, int maxRank = std::numeric_limits<int>::max())
  {
    PITTS::TensorTrain<Type> TTy(dims);
    initialRightFrame(inputs, TTy, maxRank);
    EXPECT_EQ(expectedRanks, TTy.getTTranks());
    EXPECT_NEAR(Type(0), ConstEigenMap(unfold_left(TTy.subTensor(0))).norm(), Type(0));
    for(int k = 1; k < 4; k++)
    {
      const auto core = ConstEigenMap(unfold_right(TTy.subTensor(k)));
      EXPECT_NEAR(mat::Identity(core.rows(), core.rows()), mat(core * core.adjoint()), eps);
      EXPECT_NE(PITTS::TT_Orthogonality::none, TTy.isOrthonormal(k) & PITTS::TT_Orthogonality::right);
    }
  };
  check(Inputs{std::cref(TTa)}, {2, 3, 2});
  check(Inputs{std::cref(TTa), std::cref(TTb)}, {3, 5, 5});
  check(Inputs{std::cref(TTa), std::cref(TTb)}, {2, 2, 2}, 2);
  // dependent inputs: no additional directions
  check(Inputs{std::cref(TTa), std::cref(TTa)}, {2, 3, 2});
  // independent of the norms and of the gauge of the inputs: TTb much smaller, or with its norm in the first core;
  // directions that cancel in the sum of the inputs are kept
  PITTS::TensorTrain<Type> TTsmall(dims), TTgauge(dims), TTneg(dims);
  copy(TTb, TTsmall);
  copy(TTb, TTgauge);
  copy(TTa, TTneg);
  {
    PITTS::Tensor3<Type> core;
    copy(TTneg.subTensor(1), core);
    PITTS::internal::t3_scale(Type(-1), core);
    TTneg.setSubTensor(1, std::move(core));
    copy(TTsmall.subTensor(1), core);
    PITTS::internal::t3_scale(Type(1.e-10), core);
    TTsmall.setSubTensor(1, std::move(core));
    copy(TTgauge.subTensor(0), core);
    PITTS::internal::t3_scale(Type(1.e10), core);
    TTgauge.setSubTensor(0, std::move(core));
    copy(TTgauge.subTensor(3), core);
    PITTS::internal::t3_scale(Type(1.e-10), core);
    TTgauge.setSubTensor(3, std::move(core));
  }
  check(Inputs{std::cref(TTa), std::cref(TTsmall)}, {3, 5, 5});
  check(Inputs{std::cref(TTa), std::cref(TTgauge)}, {3, 5, 5});
  check(Inputs{std::cref(TTgauge), std::cref(TTa)}, {3, 5, 5});
  check(Inputs{std::cref(TTa), std::cref(TTneg)}, {2, 3, 2});

  // invalid inputs: none, different dimensions, boundary ranks != 1
  PITTS::TensorTrain<Type> TTy(dims), TTwrong({3, 4, 3, 4});
  std::vector<PITTS::Tensor3<Type>> cores(4);
  cores[0].resize(2, 3, 2);
  cores[1].resize(2, 4, 2);
  cores[2].resize(2, 3, 2);
  cores[3].resize(2, 5, 1);
  PITTS::TensorTrain<Type> TTboundary(std::move(cores));
  EXPECT_THROW(initialRightFrame(Inputs{}, TTy), std::invalid_argument);
  EXPECT_THROW(initialRightFrame(Inputs{std::cref(TTwrong)}, TTy), std::invalid_argument);
  EXPECT_THROW(initialRightFrame(Inputs{std::cref(TTboundary)}, TTy), std::invalid_argument);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, rightIndexSetsFromFrame_random_TT)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const auto eps = blockTolerance<Type>();
  const Type maxvolTol = Type(0.05);

  const std::vector<int> dims = {3, 4, 2, 5};
  const int nDim = dims.size();
  TensorTrain TT(dims);
  TT.setTTranks({2, 3, 2});
  PITTS::internal::randomGenerator.seed(4712);
  randomize(TT);

  TensorTrain frame(dims);
  initialRightFrame({std::cref(TT)}, frame);
  const std::vector<NestedIndices> J = rightIndexSetsFromFrame(frame, maxvolTol);
  // |J_k| = rank of bond k-1 (the random TT has full ranks)
  checkRightNested(J, dims, {0, 2, 3, 2});

  // the selected columns are dominant in the frame: recompute XR_k = X_k * R_{k+1} with the cached contractions
  std::vector<NestedIndices> noLeft(nDim);
  auto Rx = defineSweepData<PITTS::Tensor3<Type>>(nDim, contraction_loop_from_left<Type>(frame, noLeft), contraction_loop_from_right<Type>(frame, J));
  Rx.update(-1, 1);
  for(int k = nDim-1; k >= 1; k--)
    EXPECT_LE(columnDominance(Rx.subTensor(k), J[k]), 1 + maxvolTol + eps);

  // the sets determine the right parts of the input
  for(int k = 1; k < nDim; k++)
    EXPECT_GT(relativeSmallestSingularValuesOfRightInterfaces(TT, J)[k], Type(1.e-3));

  // oversampled sets (at least the rank, limited per bond by maxIndices and by the number of columns)
  checkRightNested(rightIndexSetsFromFrame(frame, maxvolTol, Type(1)), dims, {0, 4, 6, 4});
  checkRightNested(rightIndexSetsFromFrame(frame, maxvolTol, Type(1), {3, 3, 3}), dims, {0, 3, 3, 3});
  EXPECT_THROW(rightIndexSetsFromFrame(frame, maxvolTol, Type(1), {3, 3}), std::invalid_argument);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, rightIndexSetsFromFrame_rank_deficient_TT)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  using Inputs = std::vector<std::reference_wrapper<const TensorTrain>>;

  // TT ranks 3 but the actual rank of bond 0 is 2 (third slice of the first core is zero)
  const std::vector<int> dims = {4, 3, 3};
  TensorTrain TT(dims, 3);
  PITTS::internal::randomGenerator.seed(4713);
  randomize(TT);
  TT.editSubTensor(0, [](PITTS::Tensor3<Type>& X){for(int j = 0; j < X.n(); j++) X(0, j, 2) = Type(0);});
  checkRightNested(indexSetsOfFrame<Type>(Inputs{std::cref(TT)}, Type(0.05)), dims, {0, 2, 3});
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, rightIndexSetsFromFrame_two_TTs)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  using Inputs = std::vector<std::reference_wrapper<const TensorTrain>>;
  const Type maxvolTol = Type(0.05);

  const std::vector<int> dims = {3, 4, 2, 5};
  TensorTrain TTa(dims), TTb(dims);
  TTa.setTTranks({2, 3, 2});
  TTb.setTTranks({1, 2, 3});
  PITTS::internal::randomGenerator.seed(4714);   // same data on all MPI processes (PITTS::initialize seeds them differently)
  randomize(TTa);
  randomize(TTb);
  const Inputs inputs = {std::cref(TTa), std::cref(TTb)};

  // |J_k| = min(r^a_{k-1} + r^b_{k-1}, n_k |J_{k+1}|)
  const std::vector<NestedIndices> J = indexSetsOfFrame(inputs, maxvolTol);
  checkRightNested(J, dims, {0, 3, 5, 5});

  // the crosses determine the right parts of both tensor trains: R_k has full row rank for both
  for(const TensorTrain& TT: {std::cref(TTa), std::cref(TTb)})
    for(int k = 1; k < int(dims.size()); k++)
      EXPECT_GT(relativeSmallestSingularValuesOfRightInterfaces(TT, J)[k], Type(1.e-3));

  // the rank of the frame limits the size of the sets
  checkRightNested(indexSetsOfFrame(inputs, maxvolTol, 2), dims, {0, 2, 2, 2});
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, rightIndexSetsFromFrame_dependent_TTs)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  using Inputs = std::vector<std::reference_wrapper<const TensorTrain>>;
  const Type maxvolTol = Type(0.05);

  // x and -2x have the same right parts: the direct sum is rank-deficient, the sets get the size of a single input
  const std::vector<int> dims = {3, 4, 2, 5};
  TensorTrain TTa(dims), TTb(dims);
  TTa.setTTranks({2, 3, 2});
  PITTS::internal::randomGenerator.seed(4715);   // same data on all MPI processes (PITTS::initialize seeds them differently)
  randomize(TTa);
  copy(TTa, TTb);
  TTb.editSubTensor(1, [](PITTS::Tensor3<Type>& X){PITTS::internal::t3_scale(Type(-2), X);});

  const std::vector<NestedIndices> J = indexSetsOfFrame(Inputs{std::cref(TTa), std::cref(TTb)}, maxvolTol);
  checkRightNested(J, dims, {0, 2, 3, 2});
  for(int k = 1; k < int(dims.size()); k++)
    EXPECT_GT(relativeSmallestSingularValuesOfRightInterfaces(TTa, J)[k], Type(1.e-3));
}
