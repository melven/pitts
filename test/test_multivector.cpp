#include <gtest/gtest.h>
#include "pitts_multivector.hpp"
#include "pitts_chunk.hpp"
#include <type_traits>

TEST(PITTS_MultiVector, type_traits)
{
  using MultiVector_double = PITTS::MultiVector<double>;

  // implicit copying is not desired
  ASSERT_FALSE(std::is_copy_constructible<MultiVector_double>());
  ASSERT_FALSE(std::is_copy_assignable<MultiVector_double>());

  // move / swap is ok
  ASSERT_TRUE(std::is_nothrow_move_constructible<MultiVector_double>());
  ASSERT_TRUE(std::is_nothrow_move_assignable<MultiVector_double>());
  ASSERT_TRUE(std::is_nothrow_swappable<MultiVector_double>());
}



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

TEST(PITTS_MultiVector, memory_layout_and_zero_padding_singleChunk)
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

TEST(PITTS_MultiVector, memory_layout_and_zero_padding_small)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  constexpr auto chunkSize = PITTS::Chunk<double>::size;

  MultiVector_double M;

  for(auto n: {1, 2, 3, 10, 20, 31, 32, 33, 50, 60, 61, 62, 63, 64, 128})
  {
    // check that the memory layout avoids strides that are powers of two
    M.resize(n, 3);
    EXPECT_EQ(M.rowChunks(), M.colStrideChunks());
    ASSERT_EQ(n, M.rows());
    ASSERT_EQ(3, M.cols());
    ASSERT_GE(M.rowChunks()*chunkSize, n);

    // check that there is zero padding directly behind the data
    for(int j = 0; j < 3; j++)
    {
      const auto iChunk = M.rowChunks() - 1;
      if( n % chunkSize == 0 )
        continue;
      for(auto ii = n % chunkSize; ii < chunkSize; ii++)
      {
        EXPECT_EQ(0., M.chunk(iChunk,j)[ii]);
      }
    }
  }
}

TEST(PITTS_MultiVector, memory_layout_and_zero_padding_large)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  constexpr auto chunkSize = PITTS::Chunk<double>::size;

  MultiVector_double M;

  for(auto n: {250, 333, 480, 496, 500, 504, 512, 520, 528, 576, 786, 1000, 1024, 1040})
  {
    // check that the memory layout avoids strides that are powers of two
    M.resize(n, 3);
    EXPECT_EQ(4, M.colStrideChunks() % 8);
    ASSERT_EQ(n, M.rows());
    ASSERT_EQ(3, M.cols());
    ASSERT_GE(M.rowChunks()*chunkSize, n);

    // check that there is zero padding directly behind the data
    for(int j = 0; j < 3; j++)
    {
      const auto iChunk = M.rowChunks() - 1;
      if( n % chunkSize == 0 )
        continue;
      for(auto ii = n % chunkSize; ii < chunkSize; ii++)
      {
        EXPECT_EQ(0., M.chunk(iChunk,j)[ii]);
      }
    }
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

TEST(PITTS_MultiVector, copy_small)
{
  constexpr auto eps = 1.e-10;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double M(3, 7);

  // set to some known constants
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      M(i,j) = 1023*i + j;

  MultiVector_double N;
  copy(M, N);

  ASSERT_EQ(3, N.rows());
  ASSERT_EQ(7, N.cols());
  ASSERT_EQ(3, M.rows());
  ASSERT_EQ(7, M.cols());
  for(int i = 0; i < 3; i++)
  {
    for(int j = 0; j < 7; j++)
    {
      ASSERT_NEAR(1023*i + j, M(i,j), eps);
      ASSERT_NEAR(1023*i + j, N(i,j), eps);
    }
  }
}

TEST(PITTS_MultiVector, copy_large)
{
  constexpr auto eps = 1.e-10;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double M(103, 3);

  // set to some known constants
  for(int i = 0; i < 103; i++)
    for(int j = 0; j < 3; j++)
      M(i,j) = 1023*i + j;

  MultiVector_double N;
  copy(M, N);

  ASSERT_EQ(103, N.rows());
  ASSERT_EQ(3, N.cols());
  ASSERT_EQ(103, M.rows());
  ASSERT_EQ(3, M.cols());
  for(int i = 0; i < 103; i++)
  {
    for(int j = 0; j < 3; j++)
    {
      ASSERT_NEAR(1023*i + j, M(i,j), eps);
      ASSERT_NEAR(1023*i + j, N(i,j), eps);
    }
  }
}
