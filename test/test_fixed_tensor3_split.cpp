#include <gtest/gtest.h>
#include "pitts_fixed_tensor3_split.hpp"
#include "pitts_fixed_tensor3_combine.hpp"

TEST(PITTS_FixedTensor3_split, N_equals_one)
{
  using FixedTensor3_double = PITTS::FixedTensor3<double,1>;
  constexpr auto eps = 1.e-10;

  FixedTensor3_double t3c(5,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      t3c(i,0,j) = 100 + i*10 + j;

  FixedTensor3_double t3a, t3b;
  split(t3c, t3a, t3b);
  const auto t3c_ = combine(t3a, t3b);

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
    {
      EXPECT_NEAR(t3c(i,0,j), t3c_(i,0,j), eps);
    }
}

TEST(PITTS_FixedTensor3_split, N_equals_four)
{
  using FixedTensor3_double4 = PITTS::FixedTensor3<double,4>;
  using FixedTensor3_double2 = PITTS::FixedTensor3<double,2>;
  constexpr auto eps = 1.e-10;

  FixedTensor3_double4 t3c(5,3);
  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
        t3c(i,k,j) = 100 + i*10 + j + k * 0.9777;

  FixedTensor3_double2 t3a, t3b;
  split(t3c, t3a, t3b);
  const auto t3c_ = combine(t3a, t3b);

  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
      {
        EXPECT_NEAR(t3c(i,k,j), t3c_(i,k,j), eps);
      }
}
