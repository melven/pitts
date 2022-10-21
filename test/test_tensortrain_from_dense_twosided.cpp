#include <gtest/gtest.h>
#include "pitts_tensortrain_from_dense_twosided.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"

namespace
{
  auto toMultiVector(const double* begin, const double* end, const std::vector<int>& dimensions)
  {
    int size = 1;
    for(auto d: dimensions)
      size *= d;
    assert(end - begin == size);
    PITTS::MultiVector<double> result(size / dimensions.back(), dimensions.back());
    for(int j = 0; j < result.cols(); j++)
      for(int i = 0; i < result.rows(); i++)
        result(i,j) = *(begin++);
    return result;
  }
}

TEST(PITTS_TensorTrain_fromDense_twoSided, scalar)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,1> scalar = {5};
  const std::vector<int> dimensions = {1};

  auto data = toMultiVector(begin(scalar), end(scalar), dimensions);
  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_NEAR(5., TT.subTensor(0)(0,0,0), eps);
}

TEST(PITTS_TensorTrain_fromDense_twoSided, vector_1d)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,7> scalar = {1,2,3,4,5,6,7};
  const std::vector<int> dimensions = {7};

  auto data = toMultiVector(begin(scalar), end(scalar), dimensions);
  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensor(0).r1());
  ASSERT_EQ(7, TT.subTensor(0).n());
  ASSERT_EQ(1, TT.subTensor(0).r2());
  ASSERT_NEAR(1., TT.subTensor(0)(0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensor(0)(0,1,0), eps);
  ASSERT_NEAR(3., TT.subTensor(0)(0,2,0), eps);
  ASSERT_NEAR(4., TT.subTensor(0)(0,3,0), eps);
  ASSERT_NEAR(5., TT.subTensor(0)(0,4,0), eps);
  ASSERT_NEAR(6., TT.subTensor(0)(0,5,0), eps);
  ASSERT_NEAR(7., TT.subTensor(0)(0,6,0), eps);
}

TEST(PITTS_TensorTrain_fromDense_twoSided, matrix_2d_1x1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,1> M = {7.};
  const std::vector<int> dimensions = {1,1};

  auto data = toMultiVector(begin(M), end(M), dimensions);
  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensor(0).r1());
  ASSERT_EQ(1, TT.subTensor(0).n());
  ASSERT_EQ(1, TT.subTensor(0).r2());
  ASSERT_NEAR(7., TT.subTensor(0)(0,0,0), eps);
  ASSERT_EQ(1, TT.subTensor(1).r1());
  ASSERT_EQ(1, TT.subTensor(1).n());
  ASSERT_EQ(1, TT.subTensor(1).r2());
  ASSERT_NEAR(1., TT.subTensor(1)(0,0,0), eps);
}

TEST(PITTS_TensorTrain_fromDense_twoSided, matrix_2d_1x5)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,5> M = {1., 2., 3., 4., 5.};
  const std::vector<int> dimensions = {1,5};

  auto data = toMultiVector(begin(M), end(M), dimensions);
  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensor(0).r1());
  ASSERT_EQ(1, TT.subTensor(0).n());
  ASSERT_EQ(1, TT.subTensor(0).r2());
  ASSERT_EQ(1, TT.subTensor(1).r1());
  ASSERT_EQ(5, TT.subTensor(1).n());
  ASSERT_EQ(1, TT.subTensor(1).r2());
  ASSERT_NEAR(1., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,1,0), eps);
  ASSERT_NEAR(3., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,2,0), eps);
  ASSERT_NEAR(4., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,3,0), eps);
  ASSERT_NEAR(5., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,4,0), eps);
}

TEST(PITTS_TensorTrain_fromDense_twoSided, matrix_2d_5x1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,5> M = {1., 2., 3., 4., 5.};
  const std::vector<int> dimensions = {5,1};

  auto data = toMultiVector(begin(M), end(M), dimensions);
  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensor(0).r1());
  ASSERT_EQ(5, TT.subTensor(0).n());
  ASSERT_EQ(1, TT.subTensor(0).r2());
  ASSERT_EQ(1, TT.subTensor(1).r1());
  ASSERT_EQ(1, TT.subTensor(1).n());
  ASSERT_EQ(1, TT.subTensor(1).r2());
  ASSERT_NEAR(1., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensor(0)(0,1,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(3., TT.subTensor(0)(0,2,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(4., TT.subTensor(0)(0,3,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(5., TT.subTensor(0)(0,4,0)*TT.subTensor(1)(0,0,0), eps);
}

TEST(PITTS_TensorTrain_fromDense_twoSided, matrix_2d_5x2_rank1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,10> M = {1., 2., 3., 4., 5., 2., 4., 6., 8., 10.};
  const std::vector<int> dimensions = {5,2};

  auto data = toMultiVector(begin(M), end(M), dimensions);
  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensor(0).r1());
  ASSERT_EQ(5, TT.subTensor(0).n());
  ASSERT_EQ(1, TT.subTensor(0).r2());
  ASSERT_EQ(1, TT.subTensor(1).r1());
  ASSERT_EQ(2, TT.subTensor(1).n());
  ASSERT_EQ(1, TT.subTensor(1).r2());
  ASSERT_NEAR(1., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensor(0)(0,1,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(3., TT.subTensor(0)(0,2,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(4., TT.subTensor(0)(0,3,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(5., TT.subTensor(0)(0,4,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,1,0), eps);
  ASSERT_NEAR(4., TT.subTensor(0)(0,1,0)*TT.subTensor(1)(0,1,0), eps);
  ASSERT_NEAR(6., TT.subTensor(0)(0,2,0)*TT.subTensor(1)(0,1,0), eps);
  ASSERT_NEAR(8., TT.subTensor(0)(0,3,0)*TT.subTensor(1)(0,1,0), eps);
  ASSERT_NEAR(10., TT.subTensor(0)(0,4,0)*TT.subTensor(1)(0,1,0), eps);
}

TEST(PITTS_TensorTrain_fromDense_twoSided, matrix_2d_2x5_rank1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,10> M = {1., 2., 2., 4., 3., 6., 4., 8., 5., 10.};
  const std::vector<int> dimensions = {2,5};

  auto data = toMultiVector(begin(M), end(M), dimensions);
  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensor(0).r1());
  ASSERT_EQ(2, TT.subTensor(0).n());
  ASSERT_EQ(1, TT.subTensor(0).r2());
  ASSERT_EQ(1, TT.subTensor(1).r1());
  ASSERT_EQ(5, TT.subTensor(1).n());
  ASSERT_EQ(1, TT.subTensor(1).r2());
  ASSERT_NEAR(1., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensor(0)(0,1,0)*TT.subTensor(1)(0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,1,0), eps);
  ASSERT_NEAR(4., TT.subTensor(0)(0,1,0)*TT.subTensor(1)(0,1,0), eps);
  ASSERT_NEAR(3., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,2,0), eps);
  ASSERT_NEAR(6., TT.subTensor(0)(0,1,0)*TT.subTensor(1)(0,2,0), eps);
  ASSERT_NEAR(4., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,3,0), eps);
  ASSERT_NEAR(8., TT.subTensor(0)(0,1,0)*TT.subTensor(1)(0,3,0), eps);
  ASSERT_NEAR(5., TT.subTensor(0)(0,0,0)*TT.subTensor(1)(0,4,0), eps);
  ASSERT_NEAR(10., TT.subTensor(0)(0,1,0)*TT.subTensor(1)(0,4,0), eps);
}

TEST(PITTS_TensorTrain_fromDense_twoSided, matrix_2d_4x5)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  std::array<double,4*5> M;
  const std::vector<int> dimensions = {4,5};
  for(int i = 0; i < 4; i++)
    for(int j = 0; j < 5; j++)
      M[i+j*4] = i + j*4;

  auto data = toMultiVector(begin(M), end(M), dimensions);
  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);

  // check result with dot products
  TensorTrain_double testTT(dimensions);
  for(int i = 0; i < 4; i++)
    for(int j = 0; j < 5; j++)
    {
      testTT.setUnit({i,j});
      EXPECT_NEAR(i+j*4., dot(testTT, TT), eps);
    }
}

TEST(PITTS_TensorTrain_fromDense_twoSided, tensor_3d_rank1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  std::array<double,3*4*5> M = {};
  const std::vector<int> dimensions = {3,4,5};
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 5; k++)
        M[i+j*3+k*3*4] = 1.;

  auto data = toMultiVector(begin(M), end(M), dimensions);
  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  std::vector<int> ones = {1,1};
  ASSERT_EQ(ones, TT.getTTranks());

  // check result with dot products
  TensorTrain_double testTT(dimensions);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 5; k++)
      {
        testTT.setUnit({i,j,k});
        EXPECT_NEAR(1., dot(testTT, TT), eps);
      }
}

TEST(PITTS_TensorTrain_fromDense_twoSided, tensor_3d_3x4x5)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  std::array<double,3*4*5> M;
  const std::vector<int> dimensions = {3,4,5};
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 5; k++)
        M[i+j*3+k*3*4] = i + j*10 + k*100;

  auto data = toMultiVector(begin(M), end(M), dimensions);
  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);

  // check result with dot products
  TensorTrain_double testTT(dimensions);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 5; k++)
      {
        testTT.setUnit({i,j,k});
        EXPECT_NEAR(i + j*10. + k*100., dot(testTT, TT), eps);
      }
}

TEST(PITTS_TensorTrain_fromDense_twoSided, tensor_5d_2x3x4x2x3_unit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  std::array<double,2*3*4*2*3> M = {};
  const std::vector<int> dimensions = {2,3,4,2,3};
  const std::vector<int> dir = {1,0,2,0,2};
  M[ dir[0] + 2*dir[1] + 2*3*dir[2] + 2*3*4*dir[3] + 2*3*4*2*dir[4] ] = 1.;

  auto data = toMultiVector(begin(M), end(M), dimensions);
  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  std::vector<int> ones = {1,1,1,1};
  ASSERT_EQ(ones, TT.getTTranks());

  TensorTrain_double refTT(dimensions);
  refTT.setUnit(dir);

  EXPECT_NEAR(1., norm2(TT), eps);
  EXPECT_NEAR(1., norm2(refTT), eps);
  EXPECT_NEAR(1., dot(TT, refTT), eps);
}

TEST(PITTS_TensorTrain_fromDense_twoSided, matrix_2d_4x5_maxRank)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  std::array<double,4*5> M;
  const std::vector<int> dimensions = {4,5};
  for(int i = 0; i < 4; i++)
    for(int j = 0; j < 5; j++)
      M[i+j*4] = (i == j ? 10.-i : 0.);

  {
    // full / exact
    auto data = toMultiVector(begin(M), end(M), dimensions);
  PITTS::MultiVector<double> work;
    TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions);

    ASSERT_EQ(TT.dimensions(), dimensions);

    // check result with dot products
    TensorTrain_double testTT(dimensions);
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 5; j++)
      {
        testTT.setUnit({i,j});
        EXPECT_NEAR((i == j ? 10.-i : 0.), dot(testTT, TT), eps);
      }
  }

  {
    // truncated
    auto data = toMultiVector(begin(M), end(M), dimensions);
  PITTS::MultiVector<double> work;
    TensorTrain_double TT = PITTS::fromDense_twoSided(data, work, dimensions, 1.e-16, 3);

    ASSERT_EQ(TT.dimensions(), dimensions);

    // check result with dot products
    TensorTrain_double testTT(dimensions);
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 5; j++)
      {
        testTT.setUnit({i,j});
        if( i == j && i < 3 )
        {
          EXPECT_NEAR(10.-i, dot(testTT, TT), eps);
        }
        else
        {
          EXPECT_NEAR(0., dot(testTT, TT), eps);
        }
      }
  }

}

TEST(PITTS_TensorTrain_fromDense_twoSided, tensor5d_random_maxRank)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using MultiVector_double = PITTS::MultiVector<double>;

  std::vector<int> shape = {2,3,4,2,3};
  MultiVector_double M(2*3*4*2, 3);
  randomize(M);

  PITTS::MultiVector<double> work;
  TensorTrain_double TT = PITTS::fromDense_twoSided(M, work, shape, 1.e-16, 2);

  for(auto r: TT.getTTranks())
  {
    ASSERT_LE(r, 2);
  }
}
