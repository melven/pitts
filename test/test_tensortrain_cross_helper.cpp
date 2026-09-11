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


// ---- block helpers ----

using PITTS::internal::tt_cross::applyToBlock;
using PITTS::internal::tt_cross::relativeDifference;

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

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, relativeDifference)
{
  using Type = typename TestFixture::Type;
  constexpr auto eps = 1.e-6;

  PITTS::Tensor3<Type> A(2, 3, 2), B(2, 3, 2), C(2, 3, 2), zero(2, 3, 2);
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
      {
        B(i, j, k) = Type(1 + i + 2*j + 6*k);
        A(i, j, k) = B(i, j, k);
        C(i, j, k) = Type(1.5) * B(i, j, k);
        zero(i, j, k) = Type(0);
      }

  EXPECT_NEAR(0, relativeDifference(A, B), eps);
  // ||1.5 B - B|| / ||B|| = 0.5
  EXPECT_NEAR(0.5, relativeDifference(C, B), eps);
  // reference zero: the absolute difference is returned
  Type normB = 0;
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
        normB += B(i, j, k)*B(i, j, k);
  EXPECT_NEAR(std::sqrt(normB), relativeDifference(B, zero), eps);

  PITTS::Tensor3<Type> wrongShape(2, 3, 3);
  EXPECT_THROW(relativeDifference(wrongShape, B), std::invalid_argument);
}


// ---- local core updates ----

#include "pitts_tensor3_fold.hpp"
#include <limits>

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

  // common invariants of a left update: selected rows are the identity, entries bounded, core*carry reconstructs B
  template<typename T>
  void checkLeftUpdate(const PITTS::Tensor3<T>& B, const CoreUpdate<T>& upd, T maxvolTol, bool exactReconstruction)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    const auto eps = blockTolerance<T>();
    const int r = upd.core.r2();
    ASSERT_EQ(B.r1(), upd.core.r1());
    ASSERT_EQ(B.n(), upd.core.n());
    ASSERT_EQ(r, upd.carry.r1());
    ASSERT_EQ(B.r2(), upd.carry.r2());
    ASSERT_EQ(r, upd.selected.size());

    const auto mapCore = ConstEigenMap(unfold_left(upd.core));
    for(int j = 0; j < r; j++)
    {
      ASSERT_LE(0, upd.selected[j]);
      ASSERT_LT(upd.selected[j], mapCore.rows());
      for(int k = 0; k < r; k++)
        EXPECT_NEAR(T(j == k ? 1 : 0), mapCore(upd.selected[j], k), eps);
    }
    EXPECT_LE(mapCore.array().abs().maxCoeff(), 1 + maxvolTol + eps);

    if( exactReconstruction )
    {
      const mat reconstructed = mapCore * ConstEigenMap(upd.carry);
      EXPECT_NEAR(ConstEigenMap(unfold_left(B)), reconstructed, eps);
    }
  }

  // common invariants of a right update: selected columns are the identity, entries bounded, carry*core reconstructs B
  template<typename T>
  void checkRightUpdate(const PITTS::Tensor3<T>& B, const CoreUpdate<T>& upd, T maxvolTol, bool exactReconstruction)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    const auto eps = blockTolerance<T>();
    const int r = upd.core.r1();
    ASSERT_EQ(B.n(), upd.core.n());
    ASSERT_EQ(B.r2(), upd.core.r2());
    ASSERT_EQ(B.r1(), upd.carry.r1());
    ASSERT_EQ(r, upd.carry.r2());
    ASSERT_EQ(r, upd.selected.size());

    const auto mapCore = ConstEigenMap(unfold_right(upd.core));
    for(int j = 0; j < r; j++)
    {
      ASSERT_LE(0, upd.selected[j]);
      ASSERT_LT(upd.selected[j], mapCore.cols());
      for(int k = 0; k < r; k++)
        EXPECT_NEAR(T(j == k ? 1 : 0), mapCore(k, upd.selected[j]), eps);
    }
    EXPECT_LE(mapCore.array().abs().maxCoeff(), 1 + maxvolTol + eps);

    if( exactReconstruction )
    {
      const mat reconstructed = ConstEigenMap(upd.carry) * mapCore;
      EXPECT_NEAR(ConstEigenMap(unfold_right(B)), reconstructed, eps);
    }
  }
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_full_rank)
{
  using Type = typename TestFixture::Type;
  const Type maxvolTol = Type(0.05);

  const auto B = randomBlock<Type>(2, 4, 3);
  const CoreUpdate<Type> upd = updateCoreLeft(B, Type(0), std::numeric_limits<int>::max(), maxvolTol);
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(3, upd.core.r2());
  checkLeftUpdate(B, upd, maxvolTol, true);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_low_rank_and_maxRank)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);

  // exact rank 2 block (r1*n = 8, r2 = 3)
  PITTS::Tensor3<Type> B(2, 4, 3);
  EigenMap(unfold_left(B)) = mat::Random(8, 2) * mat::Random(2, 3);

  const CoreUpdate<Type> upd = updateCoreLeft(B, Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol);
  EXPECT_EQ(2, upd.truncatedRank);
  EXPECT_EQ(2, upd.core.r2());
  checkLeftUpdate(B, upd, maxvolTol, true);

  // maxRank limits the rank, no exact reconstruction then
  const CoreUpdate<Type> upd1 = updateCoreLeft(B, Type(0), 1, maxvolTol);
  EXPECT_EQ(1, upd1.truncatedRank);
  EXPECT_EQ(1, upd1.core.r2());
  checkLeftUpdate(B, upd1, maxvolTol, false);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_wide_block)
{
  using Type = typename TestFixture::Type;
  const Type maxvolTol = Type(0.05);

  // r1*n = 2 < r2 = 5: the rank is limited by the number of rows
  const auto B = randomBlock<Type>(1, 2, 5);
  const CoreUpdate<Type> upd = updateCoreLeft(B, Type(0), std::numeric_limits<int>::max(), maxvolTol);
  EXPECT_EQ(2, upd.core.r2());
  checkLeftUpdate(B, upd, maxvolTol, true);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_enrichment)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);
  const auto eps = blockTolerance<Type>();

  const auto B = randomBlock<Type>(2, 4, 3);
  PITTS::Tensor2<Type> E(8, 2);
  EigenMap(E).setRandom();

  const CoreUpdate<Type> upd = updateCoreLeft<Type>(B, Type(0), std::numeric_limits<int>::max(), maxvolTol, std::cref(E));
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(5, upd.core.r2());
  // the enrichment does not change the represented block (zero rows in the carry)
  checkLeftUpdate(B, upd, maxvolTol, true);

  // the columns of the core span the enrichment directions: E - core * core^+ E = 0
  const auto mapCore = ConstEigenMap(unfold_left(upd.core));
  const mat residual = ConstEigenMap(E) - mapCore * mapCore.colPivHouseholderQr().solve(ConstEigenMap(E));
  EXPECT_NEAR(mat::Zero(8, 2), residual, eps);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_enrichment_already_in_span)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);

  const auto B = randomBlock<Type>(2, 4, 3);
  // directions inside the column space of B: nothing to add
  PITTS::Tensor2<Type> E(8, 2);
  EigenMap(E) = ConstEigenMap(unfold_left(B)) * mat::Random(3, 2);

  const CoreUpdate<Type> upd = updateCoreLeft<Type>(B, Type(0), std::numeric_limits<int>::max(), maxvolTol, std::cref(E));
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(3, upd.core.r2());
  checkLeftUpdate(B, upd, maxvolTol, true);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreLeft_enrichment_limited_by_rows)
{
  using Type = typename TestFixture::Type;
  const Type maxvolTol = Type(0.05);

  // the left unfolding has only r1*n = 4 rows: at most 4 - rank directions can be added
  const auto B = randomBlock<Type>(2, 2, 3);
  PITTS::Tensor2<Type> E(4, 2);
  EigenMap(E).setRandom();
  const CoreUpdate<Type> upd = updateCoreLeft<Type>(B, Type(0), std::numeric_limits<int>::max(), maxvolTol, std::cref(E));
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(4, upd.core.r2());
  checkLeftUpdate(B, upd, maxvolTol, true);

  // full rank already: nothing can be added (the projected remainder is only round-off)
  const auto Bfull = randomBlock<Type>(1, 3, 5);
  PITTS::Tensor2<Type> Efull(3, 2);
  EigenMap(Efull).setRandom();
  const CoreUpdate<Type> updFull = updateCoreLeft<Type>(Bfull, Type(0), std::numeric_limits<int>::max(), maxvolTol, std::cref(Efull));
  EXPECT_EQ(3, updFull.truncatedRank);
  EXPECT_EQ(3, updFull.core.r2());
  checkLeftUpdate(Bfull, updFull, maxvolTol, true);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreRight_enrichment_limited_by_columns)
{
  using Type = typename TestFixture::Type;
  const Type maxvolTol = Type(0.05);

  // full rank already (r1 = 3 = n*r2): nothing can be added
  const auto B = randomBlock<Type>(3, 3, 1);
  PITTS::Tensor2<Type> E(2, 3);
  EigenMap(E).setRandom();
  const CoreUpdate<Type> upd = updateCoreRight<Type>(B, Type(0), std::numeric_limits<int>::max(), maxvolTol, std::cref(E));
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(3, upd.core.r1());
  checkRightUpdate(B, upd, maxvolTol, true);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreRight_full_rank)
{
  using Type = typename TestFixture::Type;
  const Type maxvolTol = Type(0.05);

  const auto B = randomBlock<Type>(3, 4, 2);
  const CoreUpdate<Type> upd = updateCoreRight(B, Type(0), std::numeric_limits<int>::max(), maxvolTol);
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(3, upd.core.r1());
  checkRightUpdate(B, upd, maxvolTol, true);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreRight_low_rank_and_maxRank)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);

  PITTS::Tensor3<Type> B(3, 4, 2);
  EigenMap(unfold_right(B)) = mat::Random(3, 2) * mat::Random(2, 8);

  const CoreUpdate<Type> upd = updateCoreRight(B, Type(1.e-4), std::numeric_limits<int>::max(), maxvolTol);
  EXPECT_EQ(2, upd.truncatedRank);
  EXPECT_EQ(2, upd.core.r1());
  checkRightUpdate(B, upd, maxvolTol, true);

  const CoreUpdate<Type> upd1 = updateCoreRight(B, Type(0), 1, maxvolTol);
  EXPECT_EQ(1, upd1.core.r1());
  checkRightUpdate(B, upd1, maxvolTol, false);
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, updateCoreRight_enrichment)
{
  using Type = typename TestFixture::Type;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const Type maxvolTol = Type(0.05);
  const auto eps = blockTolerance<Type>();

  const auto B = randomBlock<Type>(3, 4, 2);
  PITTS::Tensor2<Type> E(2, 8);
  EigenMap(E).setRandom();

  const CoreUpdate<Type> upd = updateCoreRight<Type>(B, Type(0), std::numeric_limits<int>::max(), maxvolTol, std::cref(E));
  EXPECT_EQ(3, upd.truncatedRank);
  EXPECT_EQ(5, upd.core.r1());
  checkRightUpdate(B, upd, maxvolTol, true);

  // the rows of the core span the enrichment directions: E - (E core^+) core = 0
  const auto mapCore = ConstEigenMap(unfold_right(upd.core));
  const mat coreT = mapCore.transpose();
  const mat coeff = coreT.colPivHouseholderQr().solve(ConstEigenMap(E).transpose());
  const mat residual = ConstEigenMap(E) - coeff.transpose() * mapCore;
  EXPECT_NEAR(mat::Zero(2, 8), residual, eps);
}


// ---- initial index sets from the input tensor train ----

#include "pitts_tensortrain_normalize.hpp"

using PITTS::internal::tt_cross::initialRightIndexSets;

namespace
{
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
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, initialRightIndexSets_random_TT)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const auto eps = blockTolerance<Type>();
  const Type maxvolTol = Type(0.05);

  const std::vector<int> dims = {3, 4, 2, 5};
  const int nDim = dims.size();
  TensorTrain TT(dims);
  TT.setTTranks({2, 3, 2});
  randomize(TT);

  const std::vector<NestedIndices> J = initialRightIndexSets(TT, maxvolTol);
  // |J_k| = rank of bond k-1 (the random TT has full ranks)
  checkRightNested(J, dims, {0, 2, 3, 2});

  // the selected columns are dominant in the right-orthogonalized TT: recompute X_k * R_{k+1} with the interfaces
  TensorTrain TTortho(dims);
  copy(TT, TTortho);
  rightNormalize(TTortho, Type(0));
  std::vector<NestedIndices> noLeft(nDim);
  auto Rx = defineSweepData<PITTS::Tensor2<Type>>(nDim, interface_loop_from_left<Type>(TTortho, noLeft), interface_loop_from_right<Type>(TTortho, J));
  Rx.update(-1, 1);
  for(int k = nDim-1; k >= 1; k--)
  {
    PITTS::Tensor3<Type> XR;
    if( k+1 < nDim )
      PITTS::internal::normalize_contract2(TTortho.subTensor(k), Rx.subTensor(k+1), XR);
    else
      copy(TTortho.subTensor(k), XR);
    EXPECT_LE(columnDominance(XR, J[k]), 1 + maxvolTol + eps);
  }
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, initialRightIndexSets_rank_deficient_TT)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type maxvolTol = Type(0.05);

  // TT ranks 3 but the actual rank of bond 0 is 2 (third slice of the first core is zero)
  const std::vector<int> dims = {4, 3, 3};
  TensorTrain TT(dims, 3);
  randomize(TT);
  TT.editSubTensor(0, [](PITTS::Tensor3<Type>& X){for(int j = 0; j < X.n(); j++) X(0, j, 2) = Type(0);});

  const std::vector<NestedIndices> J = initialRightIndexSets(TT, maxvolTol);
  checkRightNested(J, dims, {0, 2, 3});
}

TYPED_TEST(PITTS_TensorTrain_cross_helper_interfaces, initialRightIndexSets_invalid_boundary_rank)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  std::vector<PITTS::Tensor3<Type>> cores(2);
  cores[0].resize(2, 3, 2);
  cores[1].resize(2, 3, 1);
  TensorTrain TT(std::move(cores));
  randomize(TT);
  EXPECT_THROW(initialRightIndexSets(TT, Type(0.05)), std::invalid_argument);
}
