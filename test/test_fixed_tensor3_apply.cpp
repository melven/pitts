#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_fixed_tensor3_apply.hpp"
#include "pitts_eigen.hpp"
#include <complex>

template<typename T>
class PITTS_FixedTensor3_apply : public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<double, std::complex<double>>;
TYPED_TEST_CASE(PITTS_FixedTensor3_apply, TestTypes);


TYPED_TEST(PITTS_FixedTensor3_apply, r2_equals_one)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,3>;
  using arr3 = std::array<Type,3>;
  constexpr auto eps = 1.e-10;

  FixedTensor3 t3(5,1);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      t3(i,j,0) = 100 + i*10 + j;

  const std::array<std::array<Type,3>,3> M = {arr3{1.,2.,3.},arr3{4.,5.,6.},arr3{7.,8.,9.}};

  using Matrix = Eigen::Matrix<Type,Eigen::Dynamic,Eigen::Dynamic>;
  const auto mapM = Eigen::Map<const Matrix>(&M[0][0], 3, 3);
  auto mapT3 = Eigen::Map<Matrix>(&t3(0,0,0), 5, 3);

  Matrix refResult = mapT3 * mapM.transpose();

  PITTS::apply(t3, M);

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
    {
      EXPECT_NEAR(refResult(i,j), t3(i,j,0), eps);
    }
}

TYPED_TEST(PITTS_FixedTensor3_apply, r1_equals_one)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,3>;
  using arr3 = std::array<Type,3>;
  constexpr auto eps = 1.e-10;

  FixedTensor3 t3(1,4);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 4; j++)
      t3(0,i,j) = 100 + i*10 + j;

  const std::array<std::array<Type,3>,3> M = {arr3{1.,2.,3.},arr3{4.,5.,6.},arr3{7.,8.,9.}};

  using Matrix = Eigen::Matrix<Type,Eigen::Dynamic,Eigen::Dynamic>;
  const auto mapM = Eigen::Map<const Matrix>(&M[0][0], 3, 3);
  auto mapT3 = Eigen::Map<Matrix>(&t3(0,0,0), 3, 4);

  Matrix refResult = mapM * mapT3;

  PITTS::apply(t3, M);

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 4; j++)
    {
      EXPECT_NEAR(refResult(i,j), t3(0,i,j), eps);
    }
}

TYPED_TEST(PITTS_FixedTensor3_apply, generic)
{
  using Type = TestFixture::Type;
  using FixedTensor3 = PITTS::FixedTensor3<Type,3>;
  using arr3 = std::array<Type,3>;
  constexpr auto eps = 1.e-10;

  FixedTensor3 t3(5,2);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 2; k++)
      t3(i,j,k) = 100 + i*10 + j*2 + k;

  const std::array<std::array<Type,3>,3> M = {arr3{1.,2.,3.},arr3{4.,5.,6.},arr3{7.,8.,9.}};

  using Matrix = Eigen::Matrix<Type,Eigen::Dynamic,Eigen::Dynamic>;
  const auto mapM = Eigen::Map<const Matrix>(&M[0][0], 3, 3);
  auto mapT3_1 = Eigen::Map<Matrix>(&t3(0,0,0), 5, 3);
  auto mapT3_2 = Eigen::Map<Matrix>(&t3(0,0,1), 5, 3);

  Matrix refResult_1 = mapT3_1 * mapM.transpose();
  Matrix refResult_2 = mapT3_2 * mapM.transpose();

  PITTS::apply(t3, M);

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
    {
      EXPECT_NEAR(refResult_1(i,j), t3(i,j,0), eps);
      EXPECT_NEAR(refResult_2(i,j), t3(i,j,1), eps);
    }
}

