#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_fixed_tensor3.hpp"
#include <complex>

template<typename T>
class PITTS_FixedTensor3: public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_CASE(PITTS_FixedTensor3, TestTypes);

namespace
{
  constexpr auto eps = 1.e-10;
}

TYPED_TEST(PITTS_FixedTensor3, create_large)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,200>;
  FixedTensor3 M(3,7);

  ASSERT_EQ(3, M.r1());
  ASSERT_EQ(200, M.n());
  ASSERT_EQ(7, M.r2());
}

TYPED_TEST(PITTS_FixedTensor3, create_small)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,2>;
  FixedTensor3 M(3,7);

  ASSERT_EQ(3, M.r1());
  ASSERT_EQ(2, M.n());
  ASSERT_EQ(7, M.r2());
}

TYPED_TEST(PITTS_FixedTensor3, create_empty)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,5>;
  FixedTensor3 M;

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

TYPED_TEST(PITTS_FixedTensor3, chunkSize)
{
  using Type = TestFixture::Type;
  using Chunk = PITTS::Chunk<Type>;
  using FixedTensor3 = PITTS::FixedTensor3<Type,2>;
  Chunk chunk;

  EXPECT_EQ(chunk.size, ChunkSize<FixedTensor3>::value);
}

TYPED_TEST(PITTS_FixedTensor3, resize)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,23>;
  FixedTensor3 M(3,7);

  M.resize(2,3);

  ASSERT_EQ(2, M.r1());
  ASSERT_EQ(23, M.n());
  ASSERT_EQ(3, M.r2());
}

TYPED_TEST(PITTS_FixedTensor3, memory_layout_and_zero_padding)
{
  using Type = TestFixture::Type;
  constexpr auto chunkSize = PITTS::Chunk<Type>::size;
  using FixedTensor3 = PITTS::FixedTensor3<Type,31>;
  FixedTensor3 M(2,3);

  for(int j = 0; j < 3; j++)
    for(int k = 0; k < 31; k++)
      for(int i = 0; i < 2; i++)
        M(i,k,j) = 3.;
  
  const auto nChunks = (2*3*31-1) / chunkSize + 1;
  for(int kk = 0; kk < nChunks; kk++)
  {
    for(int k = 0; k < chunkSize; k++)
    {
      const auto off = kk*chunkSize + k;
      if( off < 2*3*31 )
      {
        EXPECT_NEAR(3., (&M(0,0,0))[chunkSize*kk+k], eps);
      }
      else
      {
        // padding
        EXPECT_NEAR(0., (&M(0,0,0))[chunkSize*kk+k], eps);
      }
    }
  }
}

/*
 * from PITTS::Tensor3
TYPED_TEST(PITTS_FixedTensor3, chunkSize_small)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type>;
  FixedTensor3 M(3,2,7);

  // we expect gaps in memory that we don't use as the dimension 2 is too small to use the chunk size
  const auto arraySize = 3*2*7;
  const auto allocatedSize = std::distance(&(M(0,0,0)), &(M(2,1,6)));
  EXPECT_GT(allocatedSize, arraySize);
}
*/

TYPED_TEST(PITTS_FixedTensor3, operator_indexing_small)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,2>;
  FixedTensor3 M(3,7);

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
        EXPECT_NEAR(0, M(i,j,k), eps);
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
        EXPECT_NEAR(77, M(i,j,k), eps);
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
        EXPECT_NEAR(i*7*2+j*7+k, M(i,j,k), eps);
      }
}

TYPED_TEST(PITTS_FixedTensor3, operator_indexing_large)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,50>;
  FixedTensor3 M(3,7);

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
        EXPECT_NEAR(0, M(i,j,k), eps);
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
        EXPECT_NEAR(77, M(i,j,k), eps);
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
        EXPECT_NEAR(i*7*50+j*7+k, M(i,j,k), eps);
      }
}
