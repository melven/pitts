#include <gtest/gtest.h>
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3_random.hpp"
#include "pitts_eigen.hpp"

using namespace PITTS;

TEST(PITTS_Tensor3_unfold, unfold_left_multivector)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);

  MultiVector_double mat;
  unfold_left(t3, mat);
  ASSERT_EQ(3 * 5, mat.rows());
  ASSERT_EQ(7, mat.cols());
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 5; j++)
      for (int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3(i, j, k), mat(i + j * 3, k), eps);
      }
}

TEST(PITTS_Tensor3_unfold, unfold_right_multivector)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);

  MultiVector_double mat;
  unfold_right(t3, mat);
  ASSERT_EQ(3, mat.rows());
  ASSERT_EQ(5 * 7, mat.cols());
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 5; j++)
      for (int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3(i, j, k), mat(i, j + k * 5), eps);
      }
}

TEST(PITTS_Tensor3_unfold, unfold_left_move_Tensor2)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);
  Tensor3_double t3_copy;
  copy(t3, t3_copy);

  Tensor2_double mat = unfold_left(std::move(t3_copy));
  ASSERT_EQ(3 * 5, mat.r1());
  ASSERT_EQ(7, mat.r2());
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 5; j++)
      for (int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3(i, j, k), mat(i + j * 3, k), eps);
      }
}

TEST(PITTS_Tensor3_unfold, unfold_right_move_Tensor2)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);
  Tensor3_double t3_copy;
  copy(t3, t3_copy);

  Tensor2_double mat = unfold_right(std::move(t3_copy));
  ASSERT_EQ(3, mat.r1());
  ASSERT_EQ(5 * 7, mat.r2());
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 5; j++)
      for (int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3(i, j, k), mat(i, j + k * 5), eps);
      }
}

TEST(PITTS_Tensor3_unfold, unfold_move_multivector)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double t3(3, 5, 7);
  randomize(t3);
  Tensor3_double t3_ref;
  copy(t3, t3_ref);

  MultiVector_double mv = unfold(std::move(t3));
  ASSERT_EQ(0, t3.r1());
  ASSERT_EQ(0, t3.n());
  ASSERT_EQ(0, t3.r2());
  ASSERT_EQ(3 * 5 * 7, mv.rows());
  ASSERT_EQ(1, mv.cols());
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 5; j++)
      for (int k = 0; k < 7; k++)
      {
        EXPECT_NEAR(t3_ref(i, j, k), mv(i + j * 3 + k * 5 * 3, 0), eps);
      }
}

TEST(PITTS_Tensor3_unfold, constView_left)
{
  const int r1 = 3, n = 5, r2 = 7;

  Tensor3<double> t3(r1, n, r2);
  for (int k = 0; k < r2; k++)
    for (int j = 0; j < n; j++)
      for (int i = 0; i < r1; i++)
        t3(i, j, k) = i + j * r1 + j * r1 * n; // unique value for each entry
  const Tensor3<double> &t3Const = t3;

  ConstTensor2View<double> t2 = unfold_left(t3Const);

  // dimensions must match
  ASSERT_EQ(r1 * n, t2.r1());
  ASSERT_EQ(r2, t2.r2());
  // ASSERT_EQ(t3.reservedChunks(), t2.reservedChunks());
  //  values must match
  for (int k = 0; k < r2; k++)
    for (int j = 0; j < n; j++)
      for (int i = 0; i < r1; i++)
        EXPECT_EQ(t3(i, j, k), t2(i + j * r1, k));
}

TEST(PITTS_Tensor3_unfold, view_left)
{
  const int r1 = 3, n = 5, r2 = 7;

  Tensor3<double> t3(r1, n, r2);
  for (int k = 0; k < r2; k++)
    for (int j = 0; j < n; j++)
      for (int i = 0; i < r1; i++)
        t3(i, j, k) = i + j * r1 + j * r1 * n; // unique value for each entry

  Tensor2View<double> t2 = unfold_left(t3);

  // dimensions must match
  ASSERT_EQ(r1 * n, t2.r1());
  ASSERT_EQ(r2, t2.r2());
  // ASSERT_EQ(t3.reservedChunks(), t2.reservedChunks());
  //  values must match
  for (int k = 0; k < r2; k++)
    for (int j = 0; j < n; j++)
      for (int i = 0; i < r1; i++)
        EXPECT_EQ(t3(i, j, k), t2(i + j * r1, k));
  // changes must propogate
  t3(1, 2, 3) = -123;
  EXPECT_EQ(-123, t2(1 + 2 * r1, 3));
  t2(2 + 1 * r1, 0) = -210;
  EXPECT_EQ(-210, t3(2, 1, 0));
}

TEST(PITTS_Tensor3_unfold, constView_right)
{
  const int r1 = 3, n = 5, r2 = 7;

  Tensor3<double> t3(r1, n, r2);
  for (int k = 0; k < r2; k++)
    for (int j = 0; j < n; j++)
      for (int i = 0; i < r1; i++)
        t3(i, j, k) = i + j * r1 + j * r1 * n; // unique value for each entry
  const Tensor3<double> &t3Const = t3;

  ConstTensor2View<double> t2 = unfold_right(t3Const);

  // dimensions must match
  ASSERT_EQ(r1, t2.r1());
  ASSERT_EQ(n * r2, t2.r2());
  // ASSERT_EQ(t3.reservedChunks(), t2.reservedChunks());
  //  values must match
  for (int k = 0; k < r2; k++)
    for (int j = 0; j < n; j++)
      for (int i = 0; i < r1; i++)
        EXPECT_EQ(t3(i, j, k), t2(i, j + k * n));
}

TEST(PITTS_Tensor3_unfold, view_right)
{
  const int r1 = 3, n = 5, r2 = 7;

  Tensor3<double> t3(r1, n, r2);
  for (int k = 0; k < r2; k++)
    for (int j = 0; j < n; j++)
      for (int i = 0; i < r1; i++)
        t3(i, j, k) = i + j * r1 + j * r1 * n; // unique value for each entry

  Tensor2View<double> t2 = unfold_right(t3);

  // dimensions must match
  ASSERT_EQ(r1, t2.r1());
  ASSERT_EQ(n * r2, t2.r2());
  // ASSERT_EQ(t3.reservedChunks(), t2.reservedChunks());
  //  values must match
  for (int k = 0; k < r2; k++)
    for (int j = 0; j < n; j++)
      for (int i = 0; i < r1; i++)
        EXPECT_EQ(t3(i, j, k), t2(i, j + k * n));
  // changes must propogate
  t3(1, 2, 3) = -123;
  EXPECT_EQ(-123, t2(1, 2 + 3 * n));
  t2(2, 1 + 0 * n) = -210;
  EXPECT_EQ(-210, t3(2, 1, 0));
}