// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensor3.hpp"
#include <array>
#include <type_traits>
#include <vector>

TEST(PITTS_Tensor3, type_traits)
{
  using Tensor3_double = PITTS::Tensor3<double>;

  // implicit copying is not desired
  ASSERT_FALSE(std::is_copy_constructible<Tensor3_double>::value);
  ASSERT_FALSE(std::is_copy_assignable<Tensor3_double>::value);

  // move / swap is ok
  ASSERT_TRUE(std::is_nothrow_move_constructible<Tensor3_double>::value);
  ASSERT_TRUE(std::is_nothrow_move_assignable<Tensor3_double>::value);
  ASSERT_TRUE(std::is_nothrow_swappable<Tensor3_double>::value);
}

TEST(PITTS_Tensor3, create_large)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  Tensor3_double M(3,200,7);

  ASSERT_EQ(3, M.r1());
  ASSERT_EQ(200, M.n());
  ASSERT_EQ(7, M.r2());
}

TEST(PITTS_Tensor3, create_small)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  Tensor3_double M(3,2,7);

  ASSERT_EQ(3, M.r1());
  ASSERT_EQ(2, M.n());
  ASSERT_EQ(7, M.r2());
}

TEST(PITTS_Tensor3, create_empty)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  Tensor3_double M;

  ASSERT_EQ(0, M.r1());
  ASSERT_EQ(0, M.n());
  ASSERT_EQ(0, M.r2());
}

TEST(PITTS_Tensor3, moved_from)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  Tensor3_double M(3,2,7);

  Tensor3_double dummy(std::move(M));

  ASSERT_EQ(0, M.r1());
  ASSERT_EQ(0, M.n());
  ASSERT_EQ(0, M.r2());
  ASSERT_EQ(0, M.reservedChunks());

  M.resize(7,1,2);
  const auto dummy_reservedChunks = dummy.reservedChunks();
  dummy = std::move(M);
  // this internally swaps storage of M and dummy, so we can reuse the memory of dummy...
  ASSERT_EQ(0, M.r1());
  ASSERT_EQ(0, M.n());
  ASSERT_EQ(0, M.r2());
  ASSERT_EQ(dummy_reservedChunks, M.reservedChunks());
}

TEST(PITTS_Tensor3, resize)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  Tensor3_double M(3,23,7);

  M.resize(2,25,3);

  ASSERT_EQ(2, M.r1());
  ASSERT_EQ(25, M.n());
  ASSERT_EQ(3, M.r2());
}

TEST(PITTS_Tensor3, operator_indexing_small)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  Tensor3_double M(3,2,7);

  // Set to zero
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 2; j++)
      for(int k = 0; k < 7; k++)
        M(i,j,k) = 0;

  // check for zero
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 2; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_EQ(0, M(i,j,k));
      }

  // Set to constant
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 2; j++)
      for(int k = 0; k < 7; k++)
      M(i,j,k) = 77;

  // check
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 2; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_EQ(77, M(i,j,k));
      }

  // set to different values
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 2; j++)
      for(int k = 0; k < 7; k++)
        M(i,j,k) = i*7*2 + j*7 + k;

  // check
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 2; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_EQ(i*7*2+j*7+k, M(i,j,k));
      }
}

TEST(PITTS_Tensor3, operator_indexing_large)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  Tensor3_double M(3,50,7);

  // Set to zero
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 50; j++)
      for(int k = 0; k < 7; k++)
        M(i,j,k) = 0;

  // check for zero
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 50; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_EQ(0, M(i,j,k));
      }

  // Set to constant
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 50; j++)
      for(int k = 0; k < 7; k++)
      M(i,j,k) = 77;

  // check
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 50; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_EQ(77, M(i,j,k));
      }

  // set to different values
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 50; j++)
      for(int k = 0; k < 7; k++)
        M(i,j,k) = i*7*50 + j*7 + k;

  // check
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 50; j++)
      for(int k = 0; k < 7; k++)
      {
        EXPECT_EQ(i*7*50+j*7+k, M(i,j,k));
      }
}

TEST(PITTS_Tensor3, copyTo_copyFrom_strided)
{
  // C order (i2 fastest) in a std::vector as an array-like object, small and large (OpenMP parallel) sizes
  for(const auto& dims: {std::array<long long,3>{2, 3, 4}, std::array<long long,3>{30, 50, 80}})
  {
    // (no structured binding: clang does not support capturing it in a lambda used in an OpenMP region)
    const long long r1 = dims[0], n = dims[1], r2 = dims[2];
    PITTS::Tensor3<double> t3(r1, n, r2);
    for(long long i2 = 0; i2 < r2; i2++)
      for(long long j = 0; j < n; j++)
        for(long long i1 = 0; i1 < r1; i1++)
          t3(i1, j, i2) = i1 + 1000*j + 1000000*i2;

    std::vector<double> buffer(r1*n*r2, -1.);
    const auto cOrder = [&buffer, n, r2](long long i1, long long j, long long i2) -> double& {return buffer[(i1*n + j)*r2 + i2];};
    PITTS::internal::copyTo(t3, cOrder);
    std::vector<double> expected(r1*n*r2);
    for(long long i1 = 0; i1 < r1; i1++)
      for(long long j = 0; j < n; j++)
        for(long long i2 = 0; i2 < r2; i2++)
          expected[(i1*n + j)*r2 + i2] = i1 + 1000*j + 1000000*i2;
    EXPECT_EQ(expected, buffer);

    // and back (into a tensor of the same size)
    PITTS::Tensor3<double> t3b(r1, n, r2);
    PITTS::internal::copyFrom(cOrder, t3b);
    std::vector<double> roundTrip, original;
    for(long long i2 = 0; i2 < r2; i2++)
      for(long long j = 0; j < n; j++)
        for(long long i1 = 0; i1 < r1; i1++)
        {
          roundTrip.push_back(t3b(i1, j, i2));
          original.push_back(t3(i1, j, i2));
        }
    EXPECT_EQ(original, roundTrip);
  }
}
