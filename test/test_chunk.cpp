// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_chunk.hpp"
#include <vector>
#include <type_traits>

TEST(PITTS_Chunk, type_traits)
{
  using Chunk_double = PITTS::Chunk<double>;

  ASSERT_TRUE(std::is_trivial<Chunk_double>());
  ASSERT_TRUE(std::is_standard_layout<Chunk_double>());
}

TEST(PITTS_Chunk, check_alignment)
{
  using Chunk_double = PITTS::Chunk<double>;
  Chunk_double dummy;

  // we should have at least 8 doubles for AVX512 support
  EXPECT_GE(Chunk_double::size, 8);
  EXPECT_EQ(0, Chunk_double::size % 8);

  // check the alignment
  EXPECT_EQ(PITTS::ALIGNMENT, alignof(Chunk_double));

  // check that there is no overhead
  EXPECT_EQ(PITTS::ALIGNMENT, sizeof(Chunk_double));
  EXPECT_EQ(Chunk_double::size*sizeof(double), sizeof(Chunk_double));
}

TEST(PITTS_Chunk, check_std_vector_addressing)
{
  using Chunk_double = PITTS::Chunk<double>;
  std::vector<Chunk_double> v(3);
  EXPECT_EQ(&v[1][0], &v[0][0] + Chunk_double::size);
}

TEST(PITTS_Chunk, internal_paddedChunks)
{
  using PITTS::internal::paddedChunks;

  EXPECT_EQ(0, paddedChunks(0));
  EXPECT_EQ(1, paddedChunks(1));
  EXPECT_EQ(2, paddedChunks(2));
  EXPECT_EQ(3, paddedChunks(3));
  EXPECT_EQ(4, paddedChunks(4));
  EXPECT_EQ(5, paddedChunks(5));
  EXPECT_EQ(6, paddedChunks(6));
  EXPECT_EQ(7, paddedChunks(7));
  EXPECT_EQ(8, paddedChunks(8));
  EXPECT_EQ(9, paddedChunks(9));
  EXPECT_EQ(10, paddedChunks(10));
  EXPECT_EQ(11, paddedChunks(11));
  EXPECT_EQ(12, paddedChunks(12));
  EXPECT_EQ(13, paddedChunks(13));
  EXPECT_EQ(14, paddedChunks(14));
  EXPECT_EQ(15, paddedChunks(15));
  EXPECT_EQ(20, paddedChunks(16));
  EXPECT_EQ(20, paddedChunks(17));
  EXPECT_EQ(20, paddedChunks(18));
  EXPECT_EQ(20, paddedChunks(19));
  EXPECT_EQ(20, paddedChunks(20));
  EXPECT_EQ(28, paddedChunks(21));
  EXPECT_EQ(28, paddedChunks(22));
  EXPECT_EQ(28, paddedChunks(23));
  EXPECT_EQ(28, paddedChunks(24));
  EXPECT_EQ(28, paddedChunks(25));
  EXPECT_EQ(28, paddedChunks(26));
  EXPECT_EQ(28, paddedChunks(27));
  EXPECT_EQ(28, paddedChunks(28));
  EXPECT_EQ(36, paddedChunks(29));
  EXPECT_EQ(36, paddedChunks(30));
  EXPECT_EQ(36, paddedChunks(31));
  EXPECT_EQ(36, paddedChunks(32));
  EXPECT_EQ(508, paddedChunks(508));
  EXPECT_EQ(516, paddedChunks(509));
  EXPECT_EQ(516, paddedChunks(510));
  EXPECT_EQ(516, paddedChunks(511));
  EXPECT_EQ(516, paddedChunks(512));
  EXPECT_EQ(516, paddedChunks(513));
  EXPECT_EQ(516, paddedChunks(514));
  EXPECT_EQ(516, paddedChunks(515));
  EXPECT_EQ(516, paddedChunks(516));
  EXPECT_EQ(524, paddedChunks(517));
  EXPECT_EQ(524, paddedChunks(518));
  EXPECT_EQ(524, paddedChunks(519));
  EXPECT_EQ(524, paddedChunks(520));
}