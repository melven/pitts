#include <gtest/gtest.h>
#include "pitts_chunk.hpp"

TEST(PITTS_Chunk, check_alignment)
{
  using Chunk_double = PITTS::Chunk<double>;
  Chunk_double dummy;

  // we should have at least 8 doubles for AVX512 support
  EXPECT_GE(dummy.size(), 8);
  EXPECT_EQ(0, dummy.size() % 8);

  // check the alignment
  EXPECT_EQ(PITTS::ALIGNMENT, alignof(Chunk_double));
}
