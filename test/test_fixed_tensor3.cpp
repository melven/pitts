#include <gtest/gtest.h>
#include "pitts_fixed_tensor3.hpp"

TEST(PITTS_FixedTensor3, create_large)
{
  using FixedTensor3_double = PITTS::FixedTensor3<double,200>;
  FixedTensor3_double M(3,7);

  ASSERT_EQ(3, M.r1());
  ASSERT_EQ(200, M.n());
  ASSERT_EQ(7, M.r2());
}

TEST(PITTS_FixedTensor3, create_small)
{
  using FixedTensor3_double = PITTS::FixedTensor3<double,2>;
  FixedTensor3_double M(3,7);

  ASSERT_EQ(3, M.r1());
  ASSERT_EQ(2, M.n());
  ASSERT_EQ(7, M.r2());
}

TEST(PITTS_FixedTensor3, create_empty)
{
  using FixedTensor3_double = PITTS::FixedTensor3<double,5>;
  FixedTensor3_double M;

  ASSERT_EQ(0, M.r1());
  ASSERT_EQ(5, M.n());
  ASSERT_EQ(0, M.r2());
}


// anonymous namespace
namespace
{
// helper class for accessing the protected member chunkSize
template<class T3>
struct ChunkSize : private T3
{
  static constexpr auto value = T3::chunkSize;
};
}

TEST(PITTS_FixedTensor3, chunkSize)
{
  using Chunk_double = PITTS::Chunk<double>;
  using FixedTensor3_double = PITTS::FixedTensor3<double,2>;
  Chunk_double chunk;

  EXPECT_EQ(chunk.size, ChunkSize<FixedTensor3_double>::value);
}

TEST(PITTS_FixedTensor3, resize)
{
  using FixedTensor3_double = PITTS::FixedTensor3<double,23>;
  FixedTensor3_double M(3,7);

  M.resize(2,3);

  ASSERT_EQ(2, M.r1());
  ASSERT_EQ(23, M.n());
  ASSERT_EQ(3, M.r2());
}

TEST(PITTS_FixedTensor3, memory_layout_and_zero_padding)
{
  constexpr auto chunkSize_double = PITTS::Chunk<double>::size;
  using FixedTensor3_double = PITTS::FixedTensor3<double,31>;
  FixedTensor3_double M(2,3);

  for(int j = 0; j < 3; j++)
    for(int k = 0; k < 31; k++)
      for(int i = 0; i < 2; i++)
        M(i,k,j) = 3.;
  
  const auto nChunks = (2*3*31-1) / chunkSize_double + 1;
  for(int kk = 0; kk < nChunks; kk++)
  {
    for(int k = 0; k < chunkSize_double; k++)
    {
      const auto off = kk*chunkSize_double + k;
      if( off < 2*3*31 )
      {
        EXPECT_EQ(3., (&M(0,0,0))[chunkSize_double*kk+k]);
      }
      else
      {
        // padding
        EXPECT_EQ(0., (&M(0,0,0))[chunkSize_double*kk+k]);
      }
    }
  }
}

/*
 * from PITTS::Tensor3
TEST(PITTS_FixedTensor3, chunkSize_small)
{
  using FixedTensor3_double = PITTS::FixedTensor3<double>;
  FixedTensor3_double M(3,2,7);

  // we expect gaps in memory that we don't use as the dimension 2 is too small to use the chunk size
  const auto arraySize = 3*2*7;
  const auto allocatedSize = std::distance(&(M(0,0,0)), &(M(2,1,6)));
  EXPECT_GT(allocatedSize, arraySize);
}
*/

TEST(PITTS_FixedTensor3, operator_indexing_small)
{
  using FixedTensor3_double = PITTS::FixedTensor3<double,2>;
  FixedTensor3_double M(3,7);

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

TEST(PITTS_FixedTensor3, operator_indexing_large)
{
  using FixedTensor3_double = PITTS::FixedTensor3<double,50>;
  FixedTensor3_double M(3,7);

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
