#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_fixed_tensor3_random.hpp"
#include <unordered_set>
#include <complex>


TEST(PITTS_FixedTensor3_random, randomize_real)
{
  using FixedTensor3 = PITTS::FixedTensor3<double,20>;

  FixedTensor3 t3(3,1);

  randomize(t3);
  EXPECT_EQ(3, t3.r1());
  EXPECT_EQ(20, t3.n());
  EXPECT_EQ(1, t3.r2());

  // we expect different values between -1 and 1
  std::unordered_set<double> values;
  for(int i = 0; i < t3.r1(); i++)
    for(int j = 0; j < t3.n(); j++)
      for(int k = 0; k < t3.r2(); k++)
        values.insert(t3(i,j,k));

  EXPECT_EQ(3*20*1, values.size());
  for(const auto& v: values)
  {
    EXPECT_LE(std::abs(v), 1);
  }
}


TEST(PITTS_FixedTensor3_random, randomize_complex)
{
  using FixedTensor3 = PITTS::FixedTensor3<std::complex<double>,20>;

  FixedTensor3 t3(3,1);

  randomize(t3);
  EXPECT_EQ(3, t3.r1());
  EXPECT_EQ(20, t3.n());
  EXPECT_EQ(1, t3.r2());

  // we expect different values between -1 and 1
  std::unordered_set<double> values;
  for(int i = 0; i < t3.r1(); i++)
    for(int j = 0; j < t3.n(); j++)
      for(int k = 0; k < t3.r2(); k++)
      {
        EXPECT_LE(std::abs(t3(i,j,k)), 1);
        values.insert(t3(i,j,k).real());
        values.insert(t3(i,j,k).imag());
      }

  EXPECT_EQ( 2 * 3*20*1, values.size());
}
