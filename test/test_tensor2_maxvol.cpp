// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_tensor2_maxvol.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_type_name.hpp"
#include "eigen_test_helper.hpp"
#include <complex>
#include <set>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
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

  // helper: the selected submatrix A(I,:)
  template<typename T>
  auto subMatrix(const PITTS::Tensor2<T>& A, const std::vector<int>& I)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    const auto mapA = ConstEigenMap(A);
    const int r = I.size();
    mat Ahat(r, r);
    for(int i = 0; i < r; i++)
      Ahat.row(i) = mapA.row(I[i]);
    return Ahat;
  }

  // helper: B = A * A(I,:)^-1 (the interpolation coefficients of the cross approximation)
  //
  // Calculated with a linear solve, not with an explicit inverse: forming A(I,:)^-1 and multiplying
  // is less accurate (and Eigen's dense inverse() is just a LU solve with the identity anyway).
  template<typename T>
  auto interpolationCoefficients(const PITTS::Tensor2<T>& A, const std::vector<int>& I)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    const auto mapA = ConstEigenMap(A);
    const mat Ahat = subMatrix(A, I);
    // B = A * A(I,:)^-1  <=>  A(I,:)^H B^H = A^H
    const mat B = Eigen::PartialPivLU<mat>(Ahat.adjoint()).solve(mapA.adjoint()).adjoint();
    return B;
  }

  // helper: max |A * A(I,:)^-1| (the "dominance" measure of the selected submatrix)
  template<typename T>
  auto dominance(const PITTS::Tensor2<T>& A, const std::vector<int>& I)
  {
    return interpolationCoefficients(A, I).array().abs().maxCoeff();
  }

  // helper: print diagnostic data of a maxvol result (plain cout, for debugging sporadic CI failures)
  template<typename T>
  void printDiagnostics(const char* testName, const PITTS::Tensor2<T>& A, const std::vector<int>& I, T tol)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    const auto mapA = ConstEigenMap(A);
    const mat Ahat = subMatrix(A, I);
    const mat B = interpolationCoefficients(A, I);

    // largest entry of B and its position: if this is larger than 1+tol, maxvol stopped too early
    Eigen::Index iMax = 0, jMax = 0;
    const auto maxB = B.array().abs().maxCoeff(&iMax, &jMax);

    // the same measure with an explicit inverse (less accurate, just for comparison)
    const mat Binv = mapA * Ahat.inverse();

    const Eigen::JacobiSVD<mat> svdA(mapA), svdAhat(Ahat);
    const auto condA = svdA.singularValues()(0) / svdA.singularValues()(A.r2()-1);
    const auto condAhat = svdAhat.singularValues()(0) / svdAhat.singularValues()(A.r2()-1);

    std::cout << "[maxvol] " << testName << ", T=" << PITTS::internal::TypeName::name<T>()
              << ", m=" << A.r1() << ", r=" << A.r2() << ", tol=" << std::abs(tol) << "\n"
              << "         max|A*A(I,:)^-1| = " << maxB << " at (" << iMax << "," << jMax << ")"
              << ", with explicit inverse: " << Binv.array().abs().maxCoeff() << "\n"
              << "         cond(A) = " << condA << ", cond(A(I,:)) = " << condAhat
              << ", |det A(I,:)| = " << std::abs(Ahat.determinant()) << "\n"
              << "         I =";
    for(int i: I)
      std::cout << " " << i;
    std::cout << std::endl;
  }

  // helper: |det A(I,:)|
  template<typename T>
  auto volume(const PITTS::Tensor2<T>& A, const std::vector<int>& I)
  {
    return std::abs(subMatrix(A, I).determinant());
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

  printDiagnostics("random_tall_is_dominant", A, I, tol);

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

  printDiagnostics("random_tall_small_tolerance_is_dominant", A, I, tol);

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
