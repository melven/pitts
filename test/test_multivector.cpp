#include <gtest/gtest.h>
#include "pitts_multivector.hpp"

TEST(PITTS_MultiVector, create)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  MultiVector_double M(3,7);

  ASSERT_EQ(3, M.rows());
  ASSERT_EQ(7, M.cols());
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

TEST(PITTS_MultiVector, chunkSize)
{
  using Chunk_double = PITTS::Chunk<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  Chunk_double chunk;

  EXPECT_EQ(chunk.size, ChunkSize<MultiVector_double>::value);
}

TEST(PITTS_MultiVector, resize)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  MultiVector_double M(3,7);

  M.resize(2,3);

  ASSERT_EQ(2, M.rows());
  ASSERT_EQ(3, M.cols());
}

TEST(PITTS_MultiVector, memory_layout_and_zero_padding)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  constexpr auto chunkSize = PITTS::Chunk<double>::size;
  ASSERT_GE(chunkSize, 5);
  MultiVector_double M(5,7);
  for(int j = 0; j < 7; j++)
    for(int i = 0; i < 5; i++)
      M(i,j) = 7.;
  
  auto* Mdata = &M(0,0);
  for(int j = 0; j < 7; j++)
    for(int i = 0; i < 5; i++)
    {
      EXPECT_EQ(7., Mdata[i+j*chunkSize]);
    }
  
  for(int j = 0; j < 7; j++)
    for(int i = 5; i < chunkSize; i++)
    {
      EXPECT_EQ(0., Mdata[i+j*chunkSize]);
    }

  M.resize(1,1);
  M(0,0) = 5.;
  ASSERT_EQ(Mdata, &M(0,0));
  EXPECT_EQ(5., Mdata[0]);
  for(int k = 1; k % chunkSize != 0; k++)
  {
    EXPECT_EQ(0., Mdata[k]);
  }

  M.resize(1,2);
  M(0,0) = 1.;
  M(0,1) = 2.;
  Mdata = &M(0,0);
  ASSERT_EQ(Mdata, &M(0,0));
  EXPECT_EQ(1., Mdata[0]);
  EXPECT_EQ(2., Mdata[0+chunkSize]);
  for(int k = 1; k % chunkSize != 0; k++)
  {
    EXPECT_EQ(0., Mdata[k]);
    EXPECT_EQ(0., Mdata[k+chunkSize]);
  }
}

TEST(PITTS_MultiVector, operator_indexing)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  MultiVector_double M(3,7);

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
