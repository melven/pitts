#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_fixed_tensor3_combine.hpp"
#include "pitts_eigen.hpp"
#include <complex>

template<typename T>
class PITTS_FixedTensor3_combine : public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_CASE(PITTS_FixedTensor3_combine, TestTypes);

TYPED_TEST(PITTS_FixedTensor3_combine, N_equals_one)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,1>;
  constexpr auto eps = 1.e-10;

  FixedTensor3 t3a(5,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      t3a(i,0,j) = 100 + i*10 + j;

  FixedTensor3 t3b(3,7);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      t3b(i,0,j) = 10 + i*0.5 + j*0.337;

  using Matrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using Map = Eigen::Map<Matrix>;
  auto mapT3a = Map(&t3a(0,0,0), 5, 3);
  auto mapT3b = Map(&t3b(0,0,0), 3, 7);

  Matrix refResult = mapT3a * mapT3b;

  const auto t3c = combine(t3a, t3b, false);
  const auto t3c_swapped = combine(t3a, t3b, true);
  ASSERT_EQ(5,t3c.r1());
  ASSERT_EQ(1,t3c.n());
  ASSERT_EQ(7,t3c.r2());

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_NEAR(refResult(i,j), t3c(i,0,j), eps);
      // no difference due to transposing the second dimension
      EXPECT_NEAR(refResult(i,j), t3c_swapped(i,0,j), eps);
    }
}

TYPED_TEST(PITTS_FixedTensor3_combine, N_equals_2)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,2>;
  constexpr auto eps = 1.e-10;

  FixedTensor3 t3a(5,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
        t3a(i,k,j) = 100 + i*10 + j - k * 0.77777;

  FixedTensor3 t3b(3,7);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      for(int k = 0; k < 2; k++)
        t3b(i,k,j) = 10 + i*0.5 + j*0.337 + k * 0.027;

  using Matrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using Map = Eigen::Map<Matrix>;
  auto mapT3a = Map(&t3a(0,0,0), 10, 3);
  auto mapT3b = Map(&t3b(0,0,0), 3, 14);

  Matrix refResult = mapT3a * mapT3b;

  const auto t3c = combine(t3a, t3b);
  ASSERT_EQ(5,t3c.r1());
  ASSERT_EQ(4,t3c.n());
  ASSERT_EQ(7,t3c.r2());

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_NEAR(refResult(i+0,2*j+0), t3c(i,0,j), eps);
      EXPECT_NEAR(refResult(i+5,2*j+0), t3c(i,1,j), eps);
      EXPECT_NEAR(refResult(i+0,2*j+1), t3c(i,2,j), eps);
      EXPECT_NEAR(refResult(i+5,2*j+1), t3c(i,3,j), eps);
    }
}

TYPED_TEST(PITTS_FixedTensor3_combine, N_equals_2_swapped)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,2>;
  constexpr auto eps = 1.e-8;

  FixedTensor3 t3a(5,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
        t3a(i,k,j) = 100 + i*10 + j - k * 0.77777;

  FixedTensor3 t3b(3,7);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      for(int k = 0; k < 2; k++)
        t3b(i,k,j) = 10 + i*0.5 + j*0.337 + k * 0.027;

  using Matrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  using Map = Eigen::Map<Matrix>;
  auto mapT3a = Map(&t3a(0,0,0), 10, 3);
  auto mapT3b = Map(&t3b(0,0,0), 3, 14);

  Matrix refResult = mapT3a * mapT3b;

  const auto t3c = combine(t3a, t3b, true);
  ASSERT_EQ(5,t3c.r1());
  ASSERT_EQ(4,t3c.n());
  ASSERT_EQ(7,t3c.r2());

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_NEAR(refResult(i+0,2*j+0), t3c(i,0,j), eps);
      EXPECT_NEAR(refResult(i+5,2*j+0), t3c(i,2,j), eps);
      EXPECT_NEAR(refResult(i+0,2*j+1), t3c(i,1,j), eps);
      EXPECT_NEAR(refResult(i+5,2*j+1), t3c(i,3,j), eps);
    }
}
