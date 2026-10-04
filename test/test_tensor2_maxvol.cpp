// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_tensor2_maxvol.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"
#include <complex>
#include <set>
#include <algorithm>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace
{
  // helper: check that the selected rows are a valid, unique subset of 0..m-1 of size r
  void checkValidIndices(const std::vector<int>& I, int m, int r)
  {
    ASSERT_EQ(r, I.size());
    std::set<int> unique(I.begin(), I.end());
    ASSERT_EQ(r, unique.size());
    for(int i: I)
    {
      ASSERT_LE(0, i);
      ASSERT_LT(i, m);
    }
  }

  // helper: the selected rows A(I,:)
  template<typename T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> submatrix(const PITTS::Tensor2<T>& A, const std::vector<int>& I)
  {
    const auto mapA = ConstEigenMap(A);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Ahat(I.size(), mapA.cols());
    for(int i = 0; i < I.size(); i++)
      Ahat.row(i) = mapA.row(I[i]);
    return Ahat;
  }

  // helper: max |A * A(I,:)^-1| (the "dominance" measure of the selected submatrix)
  template<typename T>
  auto dominance(const PITTS::Tensor2<T>& A, const std::vector<int>& I)
  {
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> B = ConstEigenMap(A) * submatrix(A, I).inverse();
    return B.array().abs().maxCoeff();
  }

  // helper: |det A(I,:)|
  template<typename T>
  auto volume(const PITTS::Tensor2<T>& A, const std::vector<int>& I)
  {
    return std::abs(submatrix(A, I).determinant());
  }
}

template<typename T>
class PITTS_Tensor2_maxvol : public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_CASE(PITTS_Tensor2_maxvol, TestTypes);


TYPED_TEST(PITTS_Tensor2_maxvol, invalid_dimensions)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;

  Tensor2 A(3, 5);
  EigenMap(A).setRandom();
  EXPECT_THROW(PITTS::internal::maxvol(A), std::invalid_argument);

  Tensor2 B(0, 0);
  EXPECT_THROW(PITTS::internal::maxvol(B), std::invalid_argument);
}

TYPED_TEST(PITTS_Tensor2_maxvol, rank_deficient_throws)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;

  Tensor2 A(10, 3);
  EigenMap(A).setRandom();
  // make column 2 a copy of column 0
  for(int i = 0; i < 10; i++)
    A(i, 2) = A(i, 0);
  EXPECT_THROW(PITTS::internal::maxvol(A), std::invalid_argument);
}

TYPED_TEST(PITTS_Tensor2_maxvol, square_matrix_selects_all_rows)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;

  Tensor2 A(4, 4);
  EigenMap(A).setRandom();
  EigenMap(A) += 3 * Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>::Identity(4, 4);

  std::vector<int> I = PITTS::internal::maxvol(A);
  checkValidIndices(I, 4, 4);
  // natural order, consistent with the returned coefficients (identity)
  EXPECT_EQ((std::vector<int>{0, 1, 2, 3}), I);
}

TYPED_TEST(PITTS_Tensor2_maxvol, single_column_selects_max_abs_entry)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;

  Tensor2 A(7, 1);
  for(int i = 0; i < 7; i++)
    A(i, 0) = Type(0.1*(i+1));
  A(4, 0) = Type(-2);

  const std::vector<int> I = PITTS::internal::maxvol(A);
  ASSERT_EQ(1, I.size());
  EXPECT_EQ(4, I[0]);
}

TYPED_TEST(PITTS_Tensor2_maxvol, identity_block_is_found)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;

  // A = [ small random ; I ; small random ], the identity block has maximal volume 1
  Tensor2 A(12, 3);
  EigenMap(A) = 0.1 * mat::Random(12, 3);
  EigenMap(A).middleRows(5, 3) = mat::Identity(3, 3);

  std::vector<int> I = PITTS::internal::maxvol(A);
  checkValidIndices(I, 12, 3);
  std::sort(I.begin(), I.end());
  EXPECT_EQ((std::vector<int>{5, 6, 7}), I);
}

TYPED_TEST(PITTS_Tensor2_maxvol, random_tall_is_dominant)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;
  const auto eps = std::sqrt(std::numeric_limits<decltype(std::abs(Type()))>::epsilon());

  const int m = 60, r = 5;
  Tensor2 A(m, r);
  EigenMap(A).setRandom();

  const Type tol = Type(0.01);
  const std::vector<int> I = PITTS::internal::maxvol(A, tol);
  checkValidIndices(I, m, r);

  // all entries of A * A(I,:)^-1 must be bounded by 1+tol (up to round-off)
  EXPECT_LE(dominance(A, I), 1 + std::abs(tol) + 10*eps);
}

TYPED_TEST(PITTS_Tensor2_maxvol, random_tall_small_tolerance_is_dominant)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;
  const auto eps = std::sqrt(std::numeric_limits<decltype(std::abs(Type()))>::epsilon());

  const int m = 200, r = 8;
  Tensor2 A(m, r);
  EigenMap(A).setRandom();

  const Type tol = Type(1.e-4);
  const std::vector<int> I = PITTS::internal::maxvol(A, tol, 1000);
  checkValidIndices(I, m, r);

  EXPECT_LE(dominance(A, I), 1 + std::abs(tol) + 10*eps);
}

TYPED_TEST(PITTS_Tensor2_maxvol, volume_better_than_leading_rows)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;

  const int m = 40, r = 4;
  Tensor2 A(m, r);
  EigenMap(A).setRandom();

  const std::vector<int> I = PITTS::internal::maxvol(A);
  checkValidIndices(I, m, r);

  const std::vector<int> leadingRows = {0, 1, 2, 3};
  EXPECT_GE(volume(A, I), volume(A, leadingRows));
}

TYPED_TEST(PITTS_Tensor2_maxvol, invariant_under_row_permutation)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;

  const int m = 30, r = 3;
  Tensor2 A(m, r);
  EigenMap(A).setRandom();

  // reversed row order
  Tensor2 Arev(m, r);
  for(int i = 0; i < m; i++)
    for(int j = 0; j < r; j++)
      Arev(i, j) = A(m-1-i, j);

  std::vector<int> I = PITTS::internal::maxvol(A, Type(1.e-6), 1000);
  std::vector<int> Irev = PITTS::internal::maxvol(Arev, Type(1.e-6), 1000);
  checkValidIndices(I, m, r);
  checkValidIndices(Irev, m, r);

  // the selected submatrices must have the same volume (the dominant submatrix is not necessarily unique)
  EXPECT_NEAR(volume(A, I), volume(Arev, Irev), 1.e-3*volume(A, I));
}

TYPED_TEST(PITTS_Tensor2_maxvol, max_iterations_limits_swaps)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;

  const int m = 100, r = 6;
  Tensor2 A(m, r);
  EigenMap(A).setRandom();

  // maxIter = 0: only the initial pivot selection, still a valid index set
  const std::vector<int> I = PITTS::internal::maxvol(A, Type(1.e-8), 0);
  checkValidIndices(I, m, r);
  // and the volume is non-zero
  EXPECT_GT(volume(A, I), 0);
}

TYPED_TEST(PITTS_Tensor2_maxvol, returns_interpolation_coefficients)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const auto eps = 100*std::sqrt(std::numeric_limits<decltype(std::abs(Type()))>::epsilon());

  const int m = 50, r = 4;
  Tensor2 A(m, r);
  EigenMap(A).setRandom();

  Tensor2 AAinv;
  const std::vector<int> I = PITTS::internal::maxvol(A, Type(0.01), 100, &AAinv);
  checkValidIndices(I, m, r);
  ASSERT_EQ(m, AAinv.r1());
  ASSERT_EQ(r, AAinv.r2());

  // reference: A * A(I,:)^-1
  const mat ref = ConstEigenMap(A) * submatrix(A, I).inverse();
  EXPECT_NEAR(ref, ConstEigenMap(AAinv), eps);

  // the selected rows form the identity
  for(int j = 0; j < r; j++)
    for(int k = 0; k < r; k++)
      EXPECT_NEAR(Type(j == k ? 1 : 0), AAinv(I[j], k), eps);

  // square case: identity without any swaps
  Tensor2 S(r, r);
  EigenMap(S).setRandom();
  EigenMap(S) += 3 * mat::Identity(r, r);
  const std::vector<int> Isq = PITTS::internal::maxvol(S, Type(0.01), 100, &AAinv);
  checkValidIndices(Isq, r, r);
  EXPECT_NEAR(mat::Identity(r, r), ConstEigenMap(AAinv), eps);
}


TYPED_TEST(PITTS_Tensor2_maxvol, oversampling_invalid_arguments)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;

  Tensor2 A(10, 3);
  EigenMap(A).setRandom();
  EXPECT_THROW(PITTS::internal::maxvol<Type>(A, Type(0.05), 100, nullptr, 2), std::invalid_argument);

  // rank-deficient input is rejected before the oversampling
  for(int i = 0; i < 10; i++)
    A(i, 2) = A(i, 0);
  EXPECT_THROW(PITTS::internal::maxvol<Type>(A, Type(0.05), 100, nullptr, 5), std::invalid_argument);
}

TYPED_TEST(PITTS_Tensor2_maxvol, oversampling_contains_square_rows_and_is_limited_by_m)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;

  const int m = 20, r = 4;
  Tensor2 A(m, r);
  EigenMap(A).setRandom();
  const std::vector<int> Isquare = PITTS::internal::maxvol(A);
  EXPECT_EQ(Isquare, PITTS::internal::maxvol<Type>(A, Type(0.05), 100, nullptr, r));

  const std::vector<int> I = PITTS::internal::maxvol<Type>(A, Type(0.05), 100, nullptr, 9);
  checkValidIndices(I, m, 9);
  EXPECT_EQ(Isquare, std::vector<int>(I.begin(), I.begin() + r));

  // all rows are selected at most once, a larger request is limited to m rows
  const std::vector<int> Iall = PITTS::internal::maxvol<Type>(A, Type(0.05), 100, nullptr, 3*m);
  checkValidIndices(Iall, m, m);
}

TYPED_TEST(PITTS_Tensor2_maxvol, oversampling_adds_the_row_with_the_largest_volume)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using RealType = decltype(std::abs(Type()));

  const int m = 30, r = 3, nRows = 8;
  Tensor2 A(m, r);
  EigenMap(A).setRandom();
  // badly scaled rows: a naive choice of the first rows would be poor
  for(int i = 0; i < m; i++)
    for(int j = 0; j < r; j++)
      A(i, j) *= RealType(1) / (1 + i % 7);

  const std::vector<int> I = PITTS::internal::maxvol<Type>(A, Type(0.05), 100, nullptr, nRows);
  checkValidIndices(I, m, nRows);

  // greedy criterion at every step: det(A(I,:)^H A(I,:)) is maximal among the unselected candidates
  const auto gramDet = [&](const std::vector<int>& rows)
  {
    const mat Asub = submatrix(A, rows);
    return std::real((Asub.adjoint() * Asub).determinant());
  };
  for(int n = r; n < nRows; n++)
  {
    const std::vector<int> rows(I.begin(), I.begin() + n);
    auto chosenRows = rows;
    chosenRows.push_back(I[n]);
    const RealType chosen = gramDet(chosenRows);
    for(int i = 0; i < m; i++)
    {
      if( std::find(rows.begin(), rows.end(), i) != rows.end() )
        continue;
      auto rows1 = rows;
      rows1.push_back(i);
      EXPECT_LE(gramDet(rows1), chosen * (1 + RealType(100)*std::sqrt(std::numeric_limits<RealType>::epsilon())));
    }
  }
}

TYPED_TEST(PITTS_Tensor2_maxvol, oversampling_returns_least_squares_coefficients)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  const auto eps = 100*std::sqrt(std::numeric_limits<decltype(std::abs(Type()))>::epsilon());

  const int m = 50, r = 4, nRows = 7;
  Tensor2 A(m, r);
  EigenMap(A).setRandom();

  Tensor2 AApinv;
  const std::vector<int> I = PITTS::internal::maxvol(A, Type(0.01), 100, &AApinv, nRows);
  checkValidIndices(I, m, nRows);
  ASSERT_EQ(m, AApinv.r1());
  ASSERT_EQ(nRows, AApinv.r2());

  // reference: A * A(I,:)^+ with the pseudo-inverse from an SVD
  const mat AI = submatrix(A, I);
  const Eigen::JacobiSVD<mat> svd(AI, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const mat AIpinv = svd.matrixV() * svd.singularValues().cwiseInverse().asDiagonal() * svd.matrixU().adjoint();
  EXPECT_NEAR(mat(ConstEigenMap(A) * AIpinv), ConstEigenMap(AApinv), eps);

  // the selected rows form an orthogonal projector of rank r (instead of the identity without oversampling)
  const mat P = submatrix(AApinv, I);
  EXPECT_NEAR(P, mat(P * P), eps);
  EXPECT_NEAR(P, mat(P.adjoint()), eps);
  EXPECT_NEAR(Type(r), P.trace(), eps);
}

TYPED_TEST(PITTS_Tensor2_maxvol, oversampling_improves_the_pseudo_inverse)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;
  using mat = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using RealType = decltype(std::abs(Type()));

  // orthonormal columns with a strongly localized part: the square selection needs a large inverse
  const int m = 200, r = 20;
  mat G = mat::Random(m, r);
  G.topRows(5) *= RealType(20);
  Tensor2 A(m, r);
  EigenMap(A) = Eigen::HouseholderQR<mat>(G).householderQ() * mat::Identity(m, r);

  RealType smin = 0, previous = 0;
  for(int nRows: {r, 2*r, 4*r})
  {
    const std::vector<int> I = PITTS::internal::maxvol<Type>(A, Type(0.05), 100, nullptr, nRows);
    smin = Eigen::JacobiSVD<mat>(submatrix(A, I)).singularValues().minCoeff();
    // more rows can only increase the smallest singular value (the Gram matrix grows)
    EXPECT_GE(smin, previous * (1 - RealType(100)*std::numeric_limits<RealType>::epsilon()));
    previous = smin;
  }
  const std::vector<int> Isquare = PITTS::internal::maxvol(A);
  const RealType sminSquare = Eigen::JacobiSVD<mat>(submatrix(A, Isquare)).singularValues().minCoeff();
  // oversampling by a factor of 4 improves the conditioning clearly
  EXPECT_GT(smin, 2 * sminSquare);
}
