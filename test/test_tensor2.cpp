#include <gtest/gtest.h>
#include "pitts_tensor2.hpp"

TEST(PITTS_Tensor2, create)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  Tensor2_double M(3,7);

  ASSERT_EQ(3, M.r1());
  ASSERT_EQ(7, M.r2());
}

// anonymous namespace
namespace
{
// helper class for accessing the protected member chunkSize
template<class T2>
struct ChunkSize : private T2
{
  static constexpr auto value = T2::chunkSize;
};
}

TEST(PITTS_Tensor2, chunkSize)
{
  using Chunk_double = PITTS::Chunk<double>;
  using Tensor2_double = PITTS::Tensor2<double>;
  Chunk_double chunk;

  EXPECT_EQ(chunk.size(), ChunkSize<Tensor2_double>::value);
}

TEST(PITTS_Tensor2, resize)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  Tensor2_double M(3,7);

  M.resize(2,3);

  ASSERT_EQ(2, M.r1());
  ASSERT_EQ(3, M.r2());
}

TEST(PITTS_Tensor2, operator_indexing)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  Tensor2_double M(3,7);

  // Set to zero
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      M(i,j) = 0;

  // check for zero
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_EQ(0, M(i,j));
    }

  // Set to constant
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      M(i,j) = 77;

  // check
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_EQ(77, M(i,j));
    }

  // set to different values
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      M(i,j) = i*7+j;

  // check
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_EQ(i*7+j, M(i,j));
    }
}
