#include <gtest/gtest.h>
#include "pitts_chunk.hpp"
#include <vector>

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
