#include <gtest/gtest.h>
#include "pitts_tensortrain_random.hpp"
#include <set>

TEST(PITTS_TensorTrain_random, randomize)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;

  TensorTrain_double TT(3,5);

  TT.setTTranks({2,3});
  randomize(TT);
  EXPECT_EQ(std::vector<int>({2,3}), TT.getTTranks());

  // we expect different values between -1 and 1
  std::set<double> values;
  for(int i = 0; i < 3; i++)
  {
    const auto& subT = TT.subTensor(i);

    for(int i = 0; i < subT.r1(); i++)
      for(int j = 0; j < subT.n(); j++)
        for(int k = 0; k < subT.r2(); k++)
          values.insert(subT(i,j,k));
  }
  EXPECT_EQ(1*5*2+2*5*3+3*5*1, values.size());
  for(const auto& v: values)
  {
    EXPECT_LE(-1, v);
    EXPECT_LE(v, 1);
  }
}
