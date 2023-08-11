#include <gtest/gtest.h>
#include "test_complex_helper.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_combine.hpp"
#include "eigen_test_helper.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include <complex>

template<typename T>
class PITTS_Tensor3_split : public ::testing::Test
{
  public:
    using Type = T;
};

// std::complex does currently not work because block_TSQR doesn't support std::complex, yet
//using TestTypes = ::testing::Types<double, std::complex<double>>;
using TestTypes = ::testing::Types<double>;
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

  const auto [t3a, t3b] = split(t3c, 1, 1, false);
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

  const auto [t3a,t3b] = split(t3c, 2, 2, false, false);
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

  const auto [t3a,t3b] = split(t3c, 2, 3, false);
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

TYPED_TEST(PITTS_Tensor3_split, n_equals_one_transposed)
{
  using Type = typename TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3c(5,1,3);
  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
      t3c(i,0,j) = 100 + i*10 + j;

  const auto [t3a, t3b] = split(t3c, 1, 1, true);
  ASSERT_EQ(5, t3a.r1());
  ASSERT_EQ(1, t3a.n());
  ASSERT_EQ(t3a.r2(), t3b.r1());
  ASSERT_EQ(1, t3b.n());
  ASSERT_EQ(3, t3b.r2());
  const auto t3c_ = combine(t3a, t3b, true);

  for(int i = 0; i < 5; i++)
    for(int j = 0; j < 3; j++)
    {
      EXPECT_NEAR(t3c(i,0,j), t3c_(i,0,j), eps);
    }
}

TYPED_TEST(PITTS_Tensor3_split, n_equals_four_transposed)
{
  using Type = typename TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3c(5,4,3);
  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
        t3c(i,k,j) = 100 + i*10 + j + k * 0.9777;

  const auto [t3a,t3b] = split(t3c, 2, 2, true);
  ASSERT_EQ(5, t3a.r1());
  ASSERT_EQ(2, t3a.n());
  ASSERT_EQ(t3a.r2(), t3b.r1());
  ASSERT_EQ(2, t3b.n());
  ASSERT_EQ(3, t3b.r2());
  const auto t3c_ = combine(t3a, t3b, true);

  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
      {
        EXPECT_NEAR(t3c(i,k,j), t3c_(i,k,j), eps);
      }
}

TYPED_TEST(PITTS_Tensor3_split, n_equals_four_rightOrthog_transposed)
{
  using Type = typename TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3c(5,4,3);
  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
        t3c(i,k,j) = 100 + i*10 + j + k * 0.9777;

  const auto [t3a,t3b] = split(t3c, 2, 2, true, false);
  ASSERT_EQ(5, t3a.r1());
  ASSERT_EQ(2, t3a.n());
  ASSERT_EQ(t3a.r2(), t3b.r1());
  ASSERT_EQ(2, t3b.n());
  ASSERT_EQ(3, t3b.r2());
  const auto t3c_ = combine(t3a, t3b, true);

  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 4; k++)
      for(int j = 0; j < 3; j++)
      {
        EXPECT_NEAR(t3c(i,k,j), t3c_(i,k,j), eps);
      }
}

TYPED_TEST(PITTS_Tensor3_split, n_equals_six_transposed)
{
  using Type = typename TestFixture::Type;
  using Tensor3 = PITTS::Tensor3<Type>;
  constexpr auto eps = 1.e-10;

  Tensor3 t3c(5,6,3);
  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 6; k++)
      for(int j = 0; j < 3; j++)
        t3c(i,k,j) = 100 + i*10 + j + k * 0.9777;

  const auto [t3a,t3b] = split(t3c, 2, 3, true);
  ASSERT_EQ(5, t3a.r1());
  ASSERT_EQ(2, t3a.n());
  ASSERT_EQ(t3a.r2(), t3b.r1());
  ASSERT_EQ(3, t3b.n());
  ASSERT_EQ(3, t3b.r2());
  const auto t3c_ = combine(t3a, t3b, true);

  for(int i = 0; i < 5; i++)
    for(int k = 0; k < 6; k++)
      for(int j = 0; j < 3; j++)
      {
        EXPECT_NEAR(t3c(i,k,j), t3c_(i,k,j), eps);
      }
}

TYPED_TEST(PITTS_Tensor3_split, normalize_svd_diagonal)
{
  using Type = typename TestFixture::Type;
  using Tensor2 = PITTS::Tensor2<Type>;
  using vec = Eigen::VectorX<Type>;
  using mat = Eigen::MatrixX<Type>;
  constexpr auto eps = 1.e-10;

  Tensor2 M(20, 30);
  vec singularValues(20);
  for(int i = 0; i < 20; i++)
    singularValues(i) = i;
  EigenMap(M) = randomOrthoMatrix(20,20) * singularValues.asDiagonal() * randomOrthoMatrix(20, 30);

  const auto& mapM = ConstEigenMap(M);
  std::vector<Type> absErr = {1, 2, 3, 4, 5, 10, 20, 100};
  Type prev_err = 0;
  for(auto err: absErr)
  {
    const auto& [Q, B] = PITTS::internal::normalize_svd(M, true, err, 999, true);
    const auto& mapQ = ConstEigenMap(Q);
    const auto& mapB = ConstEigenMap(B);
    EXPECT_LE((mapM - mapQ*mapB).norm(), err + eps);
    EXPECT_GT((mapM - mapQ*mapB).norm(), prev_err - eps);
    EXPECT_NEAR(mat::Identity(mapQ.cols(), mapQ.cols()), mapQ.transpose() * mapQ, eps);
  }

  for(auto err: absErr)
  {
    const auto& [B, Qt] = PITTS::internal::normalize_svd(M, false, err, 999, true);
    const auto& mapB = ConstEigenMap(B);
    const auto& mapQt = ConstEigenMap(Qt);
    EXPECT_LE((mapM - mapB*mapQt).norm(), err + eps);
    EXPECT_GT((mapM - mapB*mapQt).norm(), prev_err - eps);
    EXPECT_NEAR(mat::Identity(mapQt.rows(),mapQt.rows()), mapQt * mapQt.transpose(), eps);
  }
}
