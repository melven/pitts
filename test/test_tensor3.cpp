#include <gtest/gtest.h>
#include "pitts_tensor3.hpp"
#include <type_traits>

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

TEST(PITTS_Tensor3, chunkSize)
{
  using Chunk_double = PITTS::Chunk<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  Chunk_double chunk;

  EXPECT_EQ(chunk.size, ChunkSize<Tensor3_double>::value);
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

TEST(PITTS_Tensor3, memory_layout_and_zero_padding)
{
  constexpr auto chunkSize_double = PITTS::Chunk<double>::size;
  using Tensor3_double = PITTS::Tensor3<double>;
  Tensor3_double M(2,31,3);

  for(int k = 0; k < 31; k++)
    for(int j = 0; j < 3; j++)
      for(int i = 0; i < 2; i++)
        M(i,k,j) = 3.;
  
  for(int kk = 0; kk < M.nChunks(); kk++)
    for(int j = 0; j < 3; j++)
      for(int i = 0; i < 2; i++)
      {
        if( kk != M.nChunks()-1 )
        {
          for(int k = 0; k < chunkSize_double; k++)
          {
            EXPECT_EQ(3., M.chunk(i,kk,j)[k]);
          }
        }
        else
        {
          const auto nPadding = chunkSize_double - 31 % chunkSize_double;
          for(int k = 0; k < chunkSize_double; k++)
          {
            if( k < chunkSize_double-nPadding )
            {
              EXPECT_EQ(3., M.chunk(i, kk, j)[k]);
            }
            else
            {
              EXPECT_EQ(0., M.chunk(i, kk, j)[k]);
            }
          }
        }
        
      }

  M.resize(2,1,3);
  M(0,0,0) = 2.;
  M(1,0,0) = 2.;
  M(0,0,1) = 2.;
  M(1,0,1) = 2.;
  M(0,0,2) = 2.;
  M(1,0,2) = 2.;
  for(int k = 0; k < chunkSize_double; k++)
  {
    double ref = (k==0) ? 2. : 0.;
    EXPECT_EQ(ref, M.chunk(0,0,0)[k]);
    EXPECT_EQ(ref, M.chunk(1,0,0)[k]);
    EXPECT_EQ(ref, M.chunk(0,0,1)[k]);
    EXPECT_EQ(ref, M.chunk(1,0,1)[k]);
    EXPECT_EQ(ref, M.chunk(0,0,2)[k]);
    EXPECT_EQ(ref, M.chunk(1,0,2)[k]);
  }
}

TEST(PITTS_Tensor3, chunkSize_small)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  Tensor3_double M(3,2,7);

  // we expect gaps in memory that we don't use as the dimension 2 is too small to use the chunk size
  const auto arraySize = 3*2*7;
  const auto allocatedSize = std::distance(&(M(0,0,0)), &(M(2,1,6)));
  EXPECT_GT(allocatedSize, arraySize);
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
