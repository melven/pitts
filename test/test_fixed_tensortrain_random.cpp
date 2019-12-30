#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_fixed_tensortrain_random.hpp"
#include <set>
#include <complex>

TEST(PITTS_FixedTensorTrain_random, randomize_real)
{
  using FixedTensorTrain = PITTS::FixedTensorTrain<double,5>;

  FixedTensorTrain TT(3);

  TT.setTTranks({2,3});
  randomize(TT);
  EXPECT_EQ(std::vector<int>({2,3}), TT.getTTranks());

  // we expect different values between -1 and 1
  std::set<double> values;
  for(const auto& subT: TT.subTensors())
  {
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

TEST(PITTS_FixedTensorTrain_random, randomize_complex)
{
  using FixedTensorTrain = PITTS::FixedTensorTrain<std::complex<double>,5>;

  FixedTensorTrain TT(3);

  TT.setTTranks({2,3});
  randomize(TT);
  EXPECT_EQ(std::vector<int>({2,3}), TT.getTTranks());

  // we expect different values between -1 and 1
  std::set<double> values;
  for(const auto& subT: TT.subTensors())
  {
    for(int i = 0; i < subT.r1(); i++)
      for(int j = 0; j < subT.n(); j++)
        for(int k = 0; k < subT.r2(); k++)
        {
          EXPECT_LE(std::abs(subT(i,j,k)), 1);
          values.insert( subT(i,j,k).real() );
          values.insert( subT(i,j,k).imag() );
        }
  }
  EXPECT_EQ(2 * ( 1*5*2+2*5*3+3*5*1 ), values.size());
}
