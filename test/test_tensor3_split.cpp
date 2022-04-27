#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_combine.hpp"
#include <complex>

template<typename T>
class PITTS_Tensor3_split : public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_CASE(PITTS_Tensor3_split, TestTypes);

TYPED_TEST(PITTS_Tensor3_split, n_equals_one)
{
  using Type = typename TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3c(5,1,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      t3c(i,0,j) = 100 + i*10 + j;

  const auto [t3a, t3b] = split(t3c, 1, 1);
  ASSERT_EQ(5, t3a.r1());
  ASSERT_EQ(1, t3a.n());
  ASSERT_EQ(t3a.r2(), t3b.r1());
  ASSERT_EQ(1, t3b.n());
  ASSERT_EQ(3, t3b.r2());
  const auto t3c_ = combine(t3a, t3b);

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
    {
      EXPECT_NEAR(t3c(i,0,j), t3c_(i,0,j), eps);
    }
}

TYPED_TEST(PITTS_Tensor3_split, n_equals_four)
{
  using Type = typename TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3c(5,4,3);
  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
        t3c(i,k,j) = 100 + i*10 + j + k * 0.9777;

  const auto [t3a,t3b] = split(t3c, 2, 2);
  ASSERT_EQ(5, t3a.r1());
  ASSERT_EQ(2, t3a.n());
  ASSERT_EQ(t3a.r2(), t3b.r1());
  ASSERT_EQ(2, t3b.n());
  ASSERT_EQ(3, t3b.r2());
  const auto t3c_ = combine(t3a, t3b);

  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
      {
        EXPECT_NEAR(t3c(i,k,j), t3c_(i,k,j), eps);
      }
}

TYPED_TEST(PITTS_Tensor3_split, n_equals_four_rightOrthog)
{
  using Type = typename TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3c(5,4,3);
  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
        t3c(i,k,j) = 100 + i*10 + j + k * 0.9777;

  const auto [t3a,t3b] = split(t3c, 2, 2, false);
  ASSERT_EQ(5, t3a.r1());
  ASSERT_EQ(2, t3a.n());
  ASSERT_EQ(t3a.r2(), t3b.r1());
  ASSERT_EQ(2, t3b.n());
  ASSERT_EQ(3, t3b.r2());
  const auto t3c_ = combine(t3a, t3b);

  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
      {
        EXPECT_NEAR(t3c(i,k,j), t3c_(i,k,j), eps);
      }
}

TYPED_TEST(PITTS_Tensor3_split, n_equals_six)
{
  using Type = typename TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3c(5,6,3);
  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 6; k++)
      for(int j = 0; j < 3; j++)
        t3c(i,k,j) = 100 + i*10 + j + k * 0.9777;

  const auto [t3a,t3b] = split(t3c, 2, 3);
  ASSERT_EQ(5, t3a.r1());
  ASSERT_EQ(2, t3a.n());
  ASSERT_EQ(t3a.r2(), t3b.r1());
  ASSERT_EQ(3, t3b.n());
  ASSERT_EQ(3, t3b.r2());
  const auto t3c_ = combine(t3a, t3b);

  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 6; k++)
      for(int j = 0; j < 3; j++)
      {
        EXPECT_NEAR(t3c(i,k,j), t3c_(i,k,j), eps);
      }
}

