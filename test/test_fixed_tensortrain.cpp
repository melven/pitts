#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_fixed_tensortrain.hpp"
#include <complex>

template<typename T>
class PITTS_FixedTensorTrain : public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_CASE(PITTS_FixedTensorTrain, TestTypes);

namespace
{
  constexpr auto eps = 1.e-10;
}

TYPED_TEST(PITTS_FixedTensorTrain, create_n_1)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,3>;

  FixedTensorTrain TT1(1);
  ASSERT_EQ(std::vector<int>({}), TT1.getTTranks());
  ASSERT_EQ(1, TT1.subTensors().size());
  ASSERT_EQ(1, TT1.subTensors()[0].r1());
  ASSERT_EQ(3, TT1.subTensors()[0].n());
  ASSERT_EQ(1, TT1.subTensors()[0].r2());
}

TYPED_TEST(PITTS_FixedTensorTrain, create_n_d)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain7 = PITTS::FixedTensorTrain<Type,7>;
  using FixedTensorTrain3 = PITTS::FixedTensorTrain<Type,3>;

  FixedTensorTrain7 TT1(3);
  ASSERT_EQ(std::vector<int>({1,1}), TT1.getTTranks());
  ASSERT_EQ(3, TT1.subTensors().size());
  for(const auto& subT: TT1.subTensors())
  {
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(7, subT.n());
    ASSERT_EQ(1, subT.r2());
  }

  FixedTensorTrain3 TT2(2,4);
  ASSERT_EQ(std::vector<int>({4}), TT2.getTTranks());
  ASSERT_EQ(2, TT2.subTensors().size());
  ASSERT_EQ(1, TT2.subTensors()[0].r1());
  ASSERT_EQ(3, TT2.subTensors()[0].n());
  ASSERT_EQ(4, TT2.subTensors()[0].r2());
  ASSERT_EQ(4, TT2.subTensors()[1].r1());
  ASSERT_EQ(3, TT2.subTensors()[1].n());
  ASSERT_EQ(1, TT2.subTensors()[1].r2());
}

TYPED_TEST(PITTS_FixedTensorTrain, setTTranks)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,7>;

  FixedTensorTrain TT(3);
  ASSERT_EQ(std::vector<int>({1,1}), TT.getTTranks());

  TT.setTTranks(2);
  ASSERT_EQ(std::vector<int>({2,2}), TT.getTTranks());
  ASSERT_EQ(3, TT.subTensors().size());
  ASSERT_EQ(1, TT.subTensors()[0].r1());
  ASSERT_EQ(7, TT.subTensors()[0].n());
  ASSERT_EQ(2, TT.subTensors()[0].r2());
  ASSERT_EQ(2, TT.subTensors()[1].r1());
  ASSERT_EQ(7, TT.subTensors()[1].n());
  ASSERT_EQ(2, TT.subTensors()[1].r2());
  ASSERT_EQ(2, TT.subTensors()[2].r1());
  ASSERT_EQ(7, TT.subTensors()[2].n());
  ASSERT_EQ(1, TT.subTensors()[2].r2());

  TT.setTTranks({3,4});
  ASSERT_EQ(std::vector<int>({3,4}), TT.getTTranks());
  ASSERT_EQ(3, TT.subTensors().size());
  ASSERT_EQ(1, TT.subTensors()[0].r1());
  ASSERT_EQ(7, TT.subTensors()[0].n());
  ASSERT_EQ(3, TT.subTensors()[0].r2());
  ASSERT_EQ(3, TT.subTensors()[1].r1());
  ASSERT_EQ(7, TT.subTensors()[1].n());
  ASSERT_EQ(4, TT.subTensors()[1].r2());
  ASSERT_EQ(4, TT.subTensors()[2].r1());
  ASSERT_EQ(7, TT.subTensors()[2].n());
  ASSERT_EQ(1, TT.subTensors()[2].r2());
}

TYPED_TEST(PITTS_FixedTensorTrain, setZero)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,4>;

  FixedTensorTrain TT(2,5);
  TT.setZero();
  // check result is zero everywhere!
  for(const auto& subT: TT.subTensors())
  {
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(4, subT.n());
    ASSERT_EQ(1, subT.r2());
    for(int i = 0; i < 4; i++)
    {
      EXPECT_NEAR(0, subT(0,i,0), eps);
    }
  }
}

TYPED_TEST(PITTS_FixedTensorTrain, setOnes)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,4>;

  FixedTensorTrain TT(2,5);
  TT.setOnes();
  // check result is zero everywhere!
  for(const auto& subT: TT.subTensors())
  {
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(4, subT.n());
    ASSERT_EQ(1, subT.r2());
    for(int i = 0; i < 4; i++)
    {
      EXPECT_NEAR(1, subT(0,i,0), eps);
    }
  }
}

TYPED_TEST(PITTS_FixedTensorTrain, setUnit)
{
  using Type = TestFixture::Type;
  using FixedTensorTrain = PITTS::FixedTensorTrain<Type,4>;

  FixedTensorTrain TT(3,5);
  const std::vector<int> unitIdx = {0,3,1};
  TT.setUnit(unitIdx);
  // check result is zero everywhere!
  for(int i = 0; i < 3; i++)
  {
    const auto& subT = TT.subTensors()[i];
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(4, subT.n());
    ASSERT_EQ(1, subT.r2());
    for(int j = 0; j < 4; j++)
    {
      Type ref = unitIdx[i] == j ? 1 : 0;
      EXPECT_NEAR(ref, subT(0,j,0), eps);
    }
  }
}

