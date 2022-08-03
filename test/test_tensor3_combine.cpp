#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_tensor3_combine.hpp"
#include <complex>
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options

template<typename T>
class PITTS_Tensor3_combine : public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_CASE(PITTS_Tensor3_combine, TestTypes);

TYPED_TEST(PITTS_Tensor3_combine, n_equals_one_small_rank1)
{
  using Type = TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3a(2,1,1);
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 1; j++)
      t3a(i,0,j) = 100 + i*10 + j;

  Tensor3 t3b(1,1,3);
  for(int i = 0; i < 1; i++)
    for(int j = 0; j < 3; j++)
      t3b(i,0,j) = 10000 + i*1000 + j*100;

  using Matrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  Matrix matT3a(2,1);
  Matrix matT3b(1,3);

  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 1; j++)
      matT3a(i,j) = t3a(i,0,j);

  for(int i = 0; i < 1; i++)
    for(int j = 0; j < 3; j++)
      matT3b(i,j) = t3b(i,0,j);

  Matrix refResult = matT3a * matT3b;

  const auto t3c = combine(t3a, t3b, false);
  const auto t3c_swapped = combine(t3a, t3b, true);
  ASSERT_EQ(2,t3c.r1());
  ASSERT_EQ(1,t3c.n());
  ASSERT_EQ(3,t3c.r2());

  ASSERT_EQ(2,t3c_swapped.r1());
  ASSERT_EQ(1,t3c_swapped.n());
  ASSERT_EQ(3,t3c_swapped.r2());


  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 3; j++)
    {
      //std::cout << "ij: " << i << " " << j << "\n";
      //std::cout << "refResult: " << refResult(i,j) << "\n";
      //std::cout << "t3c: " << t3c(i,0,j) << "\n";
      //std::cout << "t3c_swapped: " << t3c_swapped(i,0,j) << "\n";
      EXPECT_NEAR(refResult(i,j), t3c(i,0,j), eps);
      // no difference due to transposing the second dimension
      EXPECT_NEAR(refResult(i,j), t3c_swapped(i,0,j), eps);
    }
}

TYPED_TEST(PITTS_Tensor3_combine, n_equals_one)
{
  using Type = TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3a(5,1,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      t3a(i,0,j) = 100 + i*10 + j;

  Tensor3 t3b(3,1,7);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      t3b(i,0,j) = 10 + i*0.5 + j*0.337;

  using Matrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  Matrix matT3a(5,3);
  Matrix matT3b(3,7);

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      matT3a(i,j) = t3a(i,0,j);

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      matT3b(i,j) = t3b(i,0,j);

  Matrix refResult = matT3a * matT3b;

  const auto t3c = combine(t3a, t3b, false);
  const auto t3c_swapped = combine(t3a, t3b, true);
  ASSERT_EQ(5,t3c.r1());
  ASSERT_EQ(1,t3c.n());
  ASSERT_EQ(7,t3c.r2());

  ASSERT_EQ(5,t3c_swapped.r1());
  ASSERT_EQ(1,t3c_swapped.n());
  ASSERT_EQ(7,t3c_swapped.r2());

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_NEAR(refResult(i,j), t3c(i,0,j), eps);
      // no difference due to transposing the second dimension
      EXPECT_NEAR(refResult(i,j), t3c_swapped(i,0,j), eps);
    }
}

TYPED_TEST(PITTS_Tensor3_combine, n_equals_2)
{
  using Type = TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3a(5,2,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
        t3a(i,k,j) = 100 + i*10 + j - k * 0.77777;

  Tensor3 t3b(3,2,7);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      for(int k = 0; k < 2; k++)
        t3b(i,k,j) = 10 + i*0.5 + j*0.337 + k * 0.027;

  using Matrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  Matrix matT3a(10,3);
  Matrix matT3b(3,14);

  for(int i = 0; i < 10; i++)
    for(int j = 0; j < 3; j++)
      matT3a(i,j) = t3a(i%5,i/5,j);

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 14; j++)
      matT3b(i,j) = t3b(i,j%2,j/2);

  Matrix refResult = matT3a * matT3b;

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

TYPED_TEST(PITTS_Tensor3_combine, n_equals_2_swapped)
{
  using Type = TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-8;

  Tensor3 t3a(5,2,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
        t3a(i,k,j) = 100 + i*10 + j - k * 0.77777;

  Tensor3 t3b(3,2,7);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      for(int k = 0; k < 2; k++)
        t3b(i,k,j) = 10 + i*0.5 + j*0.337 + k * 0.027;

  using Matrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  Matrix matT3a(10,3);
  Matrix matT3b(3,14);

  for(int i = 0; i < 10; i++)
    for(int j = 0; j < 3; j++)
      matT3a(i,j) = t3a(i%5,i/5,j);

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 14; j++)
      matT3b(i,j) = t3b(i,j%2,j/2);

  Matrix refResult = matT3a * matT3b;

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

TYPED_TEST(PITTS_Tensor3_combine, n_equals_2_3)
{
  using Type = TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3a(5,2,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
        t3a(i,k,j) = 100 + i*10 + j - k * 0.77777;

  Tensor3 t3b(3,3,7);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      for(int k = 0; k < 3; k++)
        t3b(i,k,j) = 10 + i*0.5 + j*0.337 + k * 0.027;

  using Matrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  Matrix matT3a(10,3);
  Matrix matT3b(3,21);

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
        matT3a(i+k*5,j) = t3a(i,k,j);

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      for(int k = 0; k < 3; k++)
        matT3b(i,j*3+k) = t3b(i,k,j);

  Matrix refResult = matT3a * matT3b;

  const auto t3c = combine(t3a, t3b);
  ASSERT_EQ(5,t3c.r1());
  ASSERT_EQ(6,t3c.n());
  ASSERT_EQ(7,t3c.r2());

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_NEAR(refResult(i+0,3*j+0), t3c(i,0,j), eps);
      EXPECT_NEAR(refResult(i+5,3*j+0), t3c(i,1,j), eps);
      EXPECT_NEAR(refResult(i+0,3*j+1), t3c(i,2,j), eps);
      EXPECT_NEAR(refResult(i+5,3*j+1), t3c(i,3,j), eps);
      EXPECT_NEAR(refResult(i+0,3*j+2), t3c(i,4,j), eps);
      EXPECT_NEAR(refResult(i+5,3*j+2), t3c(i,5,j), eps);
    }
}

TYPED_TEST(PITTS_Tensor3_combine, n_equals_2_3_swapped)
{
  using Type = TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-8;

  Tensor3 t3a(5,2,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
        t3a(i,k,j) = 100 + i*10 + j - k * 0.77777;

  Tensor3 t3b(3,3,7);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      for(int k = 0; k < 3; k++)
        t3b(i,k,j) = 10 + i*0.5 + j*0.337 + k * 0.027;

  using Matrix = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>;
  Matrix matT3a(10,3);
  Matrix matT3b(3,21);

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
      matT3a(i+k*5,j) = t3a(i,k,j);

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      for(int k = 0; k < 3; k++)
        matT3b(i,k+3*j) = t3b(i,k,j);

  Matrix refResult = matT3a * matT3b;

  const auto t3c = combine(t3a, t3b, true);
  ASSERT_EQ(5,t3c.r1());
  ASSERT_EQ(6,t3c.n());
  ASSERT_EQ(7,t3c.r2());

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_NEAR(refResult(i+0,3*j+0), t3c(i,0,j), eps);
      EXPECT_NEAR(refResult(i+5,3*j+0), t3c(i,3,j), eps);
      EXPECT_NEAR(refResult(i+0,3*j+1), t3c(i,1,j), eps);
      EXPECT_NEAR(refResult(i+5,3*j+1), t3c(i,4,j), eps);
      EXPECT_NEAR(refResult(i+0,3*j+2), t3c(i,2,j), eps);
      EXPECT_NEAR(refResult(i+5,3*j+2), t3c(i,5,j), eps);
    }
}
