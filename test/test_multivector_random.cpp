#include <gtest/gtest.h>
#include "pitts_multivector_random.hpp"
#include <set>

TEST(PITTS_MultiVector_random, randomize)
{
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double t2(3,20);

  randomize(t2);
  EXPECT_EQ(3, t2.rows());
  EXPECT_EQ(20, t2.cols());

  // we expect different values between -1 and 1
  std::set<double> values;
  for(int i = 0; i < t2.rows(); i++)
    for(int j = 0; j < t2.cols(); j++)
        values.insert(t2(i,j));

  EXPECT_EQ(20*3, values.size());
  for(const auto& v: values)
  {
    EXPECT_LE(-1, v);
    EXPECT_LE(v, 1);
  }
}
