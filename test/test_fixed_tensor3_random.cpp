#include <gtest/gtest.h>
#include "pitts_fixed_tensor3_random.hpp"
#include <set>

TEST(PITTS_FixedTensor3_random, randomize)
{
  using FixedTensor3_double = PITTS::FixedTensor3<double,20>;

  FixedTensor3_double t3(3,1);

  randomize(t3);
  EXPECT_EQ(3, t3.r1());
  EXPECT_EQ(20, t3.n());
  EXPECT_EQ(1, t3.r2());

  // we expect different values between -1 and 1
  std::set<double> values;
  for(int i = 0; i < t3.r1(); i++)
    for(int j = 0; j < t3.n(); j++)
      for(int k = 0; k < t3.r2(); k++)
        values.insert(t3(i,j,k));

  EXPECT_EQ(3*20*1, values.size());
  for(const auto& v: values)
  {
    EXPECT_LE(-1, v);
    EXPECT_LE(v, 1);
  }
}
