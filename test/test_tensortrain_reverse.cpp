// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_reverse.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensor3_random.hpp"
#include "pitts_random.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
  // evaluate a tensor train at one multi-index: the scalar product with the corresponding unit tensor
  template<typename T>
  T evaluate(const PITTS::TensorTrain<T>& TT, const std::vector<int>& multiIndex)
  {
    PITTS::TensorTrain<T> unit(TT.dimensions());
    unit.setUnit(multiIndex);
    return dot(unit, TT);
  }

  // all multi-indices of the given dimensions
  std::vector<std::vector<int>> allMultiIndices(const std::vector<int>& dims)
  {
    std::vector<std::vector<int>> result(1);
    for(int n: dims)
    {
      std::vector<std::vector<int>> next;
      for(const auto& prefix: result)
        for(int i = 0; i < n; i++)
        {
          next.push_back(prefix);
          next.back().push_back(i);
        }
      result = std::move(next);
    }
    return result;
  }
}

template<typename T>
class PITTS_TensorTrain_reverse : public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PITTS_TensorTrain_reverse, TestTypes);

TYPED_TEST(PITTS_TensorTrain_reverse, t3_transpose)
{
  using Type = typename TestFixture::Type;

  PITTS::Tensor3<Type> A(2, 3, 4), B;
  randomize(A);
  PITTS::internal::t3_transpose(A, B);
  ASSERT_EQ(4, B.r1());
  ASSERT_EQ(3, B.n());
  ASSERT_EQ(2, B.r2());
  for(int k = 0; k < 2; k++)
    for(int i = 0; i < 3; i++)
      for(int j = 0; j < 4; j++)
        EXPECT_EQ(A(k, i, j), B(j, i, k));
}

TYPED_TEST(PITTS_TensorTrain_reverse, entries_and_ranks)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = 10*std::numeric_limits<Type>::epsilon();

  const std::vector<int> dims = {3, 4, 2, 5};
  TensorTrain TTx(dims);
  TTx.setTTranks({2, 3, 2});
  randomize(TTx);

  TensorTrain TTy(dims);
  reverse(TTx, TTy);
  ASSERT_EQ(std::vector<int>({5, 2, 4, 3}), TTy.dimensions());
  EXPECT_EQ(std::vector<int>({2, 3, 2}), TTy.getTTranks());
  for(const std::vector<int>& index: allMultiIndices(dims))
  {
    std::vector<int> reversed(index.rbegin(), index.rend());
    const Type ref = evaluate(TTx, index);
    EXPECT_NEAR(ref, evaluate(TTy, reversed), eps*(1 + std::abs(ref)));
  }

  // reversing twice gives the original cores, also in place
  reverse(TTy, TTy);
  ASSERT_EQ(dims, TTy.dimensions());
  for(int k = 0; k < 4; k++)
  {
    const auto& X = TTx.subTensor(k);
    const auto& Y = TTy.subTensor(k);
    ASSERT_EQ(X.r1(), Y.r1());
    ASSERT_EQ(X.r2(), Y.r2());
    for(int i = 0; i < X.r1(); i++)
      for(int j = 0; j < X.n(); j++)
        for(int l = 0; l < X.r2(); l++)
          EXPECT_EQ(X(i, j, l), Y(i, j, l));
  }
}

TYPED_TEST(PITTS_TensorTrain_reverse, boundary_ranks)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  // boundary ranks 2 and 3 are swapped as well
  std::vector<PITTS::Tensor3<Type>> cores(2);
  cores[0].resize(2, 3, 4);
  cores[1].resize(4, 5, 3);
  randomize(cores[0]);
  randomize(cores[1]);
  const TensorTrain TTx(std::move(cores));
  TensorTrain TTy(TTx.dimensions());
  reverse(TTx, TTy);
  EXPECT_EQ(3, TTy.subTensor(0).r1());
  EXPECT_EQ(4, TTy.subTensor(0).r2());
  EXPECT_EQ(2, TTy.subTensor(1).r2());
}

TYPED_TEST(PITTS_TensorTrain_reverse, orthogonality)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  const std::vector<int> dims = {3, 4, 2, 5};
  TensorTrain TTx(dims, 2);
  randomize(TTx);

  // a left-orthogonal TT becomes right-orthogonal and vice versa
  leftNormalize(TTx, Type(0));
  ASSERT_EQ(PITTS::TT_Orthogonality::left, TTx.isOrthogonal());
  TensorTrain TTy(dims);
  reverse(TTx, TTy);
  EXPECT_EQ(PITTS::TT_Orthogonality::right, TTy.isOrthogonal());
  for(int k = 0; k < 4; k++)
  {
    const bool left = (TTx.isOrthonormal(k) & PITTS::TT_Orthogonality::left) != PITTS::TT_Orthogonality::none;
    const bool right = (TTy.isOrthonormal(3-k) & PITTS::TT_Orthogonality::right) != PITTS::TT_Orthogonality::none;
    EXPECT_EQ(left, right);
  }

  rightNormalize(TTx, Type(0));
  reverse(TTx, TTy);
  EXPECT_EQ(PITTS::TT_Orthogonality::left, TTy.isOrthogonal());
}
