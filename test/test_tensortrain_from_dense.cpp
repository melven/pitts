#include <gtest/gtest.h>
#include "pitts_parallel.hpp"
#include "pitts_tensortrain_from_dense.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor3_combine.hpp"
#include <Eigen/Dense>

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

TEST(PITTS_TensorTrain_fromDense, scalar)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,1> scalar = {5};
  const std::vector<int> dimensions = {1};

  PITTS::MultiVector<double> work, data = toMultiVector(begin(scalar), end(scalar), dimensions);
  TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_NEAR(5., TT.subTensors()[0](0,0,0), eps);
}

TEST(PITTS_TensorTrain_fromDense, vector_1d)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,7> scalar = {1,2,3,4,5,6,7};
  const std::vector<int> dimensions = {7};

  PITTS::MultiVector<double> work, data = toMultiVector(begin(scalar), end(scalar), dimensions);
  TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensors()[0].r1());
  ASSERT_EQ(7, TT.subTensors()[0].n());
  ASSERT_EQ(1, TT.subTensors()[0].r2());
  ASSERT_NEAR(1., TT.subTensors()[0](0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensors()[0](0,1,0), eps);
  ASSERT_NEAR(3., TT.subTensors()[0](0,2,0), eps);
  ASSERT_NEAR(4., TT.subTensors()[0](0,3,0), eps);
  ASSERT_NEAR(5., TT.subTensors()[0](0,4,0), eps);
  ASSERT_NEAR(6., TT.subTensors()[0](0,5,0), eps);
  ASSERT_NEAR(7., TT.subTensors()[0](0,6,0), eps);
}

TEST(PITTS_TensorTrain_fromDense, matrix_2d_1x1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,1> M = {7.};
  const std::vector<int> dimensions = {1,1};

  PITTS::MultiVector<double> work, data = toMultiVector(begin(M), end(M), dimensions);
  TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensors()[0].r1());
  ASSERT_EQ(1, TT.subTensors()[0].n());
  ASSERT_EQ(1, TT.subTensors()[0].r2());
  ASSERT_EQ(1, TT.subTensors()[1].r1());
  ASSERT_EQ(1, TT.subTensors()[1].n());
  ASSERT_EQ(1, TT.subTensors()[1].r2());
  if( TT.subTensors()[0](0,0,0) > 0 )
  {
    ASSERT_NEAR(7., TT.subTensors()[0](0,0,0), eps);
    ASSERT_NEAR(1., TT.subTensors()[1](0,0,0), eps);
  }
  else
  {
    // sign flip is ok
    ASSERT_NEAR(-7., TT.subTensors()[0](0,0,0), eps);
    ASSERT_NEAR(-1., TT.subTensors()[1](0,0,0), eps);
  }
}

TEST(PITTS_TensorTrain_fromDense, matrix_2d_1x5)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,5> M = {1., 2., 3., 4., 5.};
  const std::vector<int> dimensions = {1,5};

  PITTS::MultiVector<double> work, data = toMultiVector(begin(M), end(M), dimensions);
  TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensors()[0].r1());
  ASSERT_EQ(1, TT.subTensors()[0].n());
  ASSERT_EQ(1, TT.subTensors()[0].r2());
  ASSERT_EQ(1, TT.subTensors()[1].r1());
  ASSERT_EQ(5, TT.subTensors()[1].n());
  ASSERT_EQ(1, TT.subTensors()[1].r2());
  ASSERT_NEAR(1., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,1,0), eps);
  ASSERT_NEAR(3., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,2,0), eps);
  ASSERT_NEAR(4., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,3,0), eps);
  ASSERT_NEAR(5., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,4,0), eps);
}

TEST(PITTS_TensorTrain_fromDense, matrix_2d_5x1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,5> M = {1., 2., 3., 4., 5.};
  const std::vector<int> dimensions = {5,1};

  PITTS::MultiVector<double> work, data = toMultiVector(begin(M), end(M), dimensions);
  TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensors()[0].r1());
  ASSERT_EQ(5, TT.subTensors()[0].n());
  ASSERT_EQ(1, TT.subTensors()[0].r2());
  ASSERT_EQ(1, TT.subTensors()[1].r1());
  ASSERT_EQ(1, TT.subTensors()[1].n());
  ASSERT_EQ(1, TT.subTensors()[1].r2());
  ASSERT_NEAR(1., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensors()[0](0,1,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(3., TT.subTensors()[0](0,2,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(4., TT.subTensors()[0](0,3,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(5., TT.subTensors()[0](0,4,0)*TT.subTensors()[1](0,0,0), eps);
}

TEST(PITTS_TensorTrain_fromDense, matrix_2d_5x2_rank1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,10> M = {1., 2., 3., 4., 5., 2., 4., 6., 8., 10.};
  const std::vector<int> dimensions = {5,2};

  PITTS::MultiVector<double> work, data = toMultiVector(begin(M), end(M), dimensions);
  TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensors()[0].r1());
  ASSERT_EQ(5, TT.subTensors()[0].n());
  ASSERT_EQ(1, TT.subTensors()[0].r2());
  ASSERT_EQ(1, TT.subTensors()[1].r1());
  ASSERT_EQ(2, TT.subTensors()[1].n());
  ASSERT_EQ(1, TT.subTensors()[1].r2());
  ASSERT_NEAR(1., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensors()[0](0,1,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(3., TT.subTensors()[0](0,2,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(4., TT.subTensors()[0](0,3,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(5., TT.subTensors()[0](0,4,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,1,0), eps);
  ASSERT_NEAR(4., TT.subTensors()[0](0,1,0)*TT.subTensors()[1](0,1,0), eps);
  ASSERT_NEAR(6., TT.subTensors()[0](0,2,0)*TT.subTensors()[1](0,1,0), eps);
  ASSERT_NEAR(8., TT.subTensors()[0](0,3,0)*TT.subTensors()[1](0,1,0), eps);
  ASSERT_NEAR(10., TT.subTensors()[0](0,4,0)*TT.subTensors()[1](0,1,0), eps);
}

TEST(PITTS_TensorTrain_fromDense, matrix_2d_2x5_rank1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::array<double,10> M = {1., 2., 2., 4., 3., 6., 4., 8., 5., 10.};
  const std::vector<int> dimensions = {2,5};

  PITTS::MultiVector<double> work, data = toMultiVector(begin(M), end(M), dimensions);
  TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  ASSERT_EQ(1, TT.subTensors()[0].r1());
  ASSERT_EQ(2, TT.subTensors()[0].n());
  ASSERT_EQ(1, TT.subTensors()[0].r2());
  ASSERT_EQ(1, TT.subTensors()[1].r1());
  ASSERT_EQ(5, TT.subTensors()[1].n());
  ASSERT_EQ(1, TT.subTensors()[1].r2());
  ASSERT_NEAR(1., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensors()[0](0,1,0)*TT.subTensors()[1](0,0,0), eps);
  ASSERT_NEAR(2., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,1,0), eps);
  ASSERT_NEAR(4., TT.subTensors()[0](0,1,0)*TT.subTensors()[1](0,1,0), eps);
  ASSERT_NEAR(3., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,2,0), eps);
  ASSERT_NEAR(6., TT.subTensors()[0](0,1,0)*TT.subTensors()[1](0,2,0), eps);
  ASSERT_NEAR(4., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,3,0), eps);
  ASSERT_NEAR(8., TT.subTensors()[0](0,1,0)*TT.subTensors()[1](0,3,0), eps);
  ASSERT_NEAR(5., TT.subTensors()[0](0,0,0)*TT.subTensors()[1](0,4,0), eps);
  ASSERT_NEAR(10., TT.subTensors()[0](0,1,0)*TT.subTensors()[1](0,4,0), eps);
}

TEST(PITTS_TensorTrain_fromDense, matrix_2d_4x5)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  std::array<double,4*5> M;
  const std::vector<int> dimensions = {4,5};
  for(int i = 0; i < 4; i++)
    for(int j = 0; j < 5; j++)
      M[i+j*4] = i + j*4;

  PITTS::MultiVector<double> work, data = toMultiVector(begin(M), end(M), dimensions);
  TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

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

TEST(PITTS_TensorTrain_fromDense, tensor_3d_rank1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  std::array<double,3*4*5> M = {};
  const std::vector<int> dimensions = {3,4,5};
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 5; k++)
        M[i+j*3+k*3*4] = 1.;

  PITTS::MultiVector<double> work, data = toMultiVector(begin(M), end(M), dimensions);
  TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

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

TEST(PITTS_TensorTrain_fromDense, tensor_3d_3x4x5)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  std::array<double,3*4*5> M;
  const std::vector<int> dimensions = {3,4,5};
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 5; k++)
        M[i+j*3+k*3*4] = i + j*10 + k*100;

  PITTS::MultiVector<double> work, data = toMultiVector(begin(M), end(M), dimensions);
  TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

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

TEST(PITTS_TensorTrain_fromDense, tensor_5d_2x3x4x2x3_unit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  std::array<double,2*3*4*2*3> M = {};
  const std::vector<int> dimensions = {2,3,4,2,3};
  const std::vector<int> dir = {1,0,2,0,2};
  M[ dir[0] + 2*dir[1] + 2*3*dir[2] + 2*3*4*dir[3] + 2*3*4*2*dir[4] ] = 1.;

  PITTS::MultiVector<double> work, data = toMultiVector(begin(M), end(M), dimensions);
  TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

  ASSERT_EQ(TT.dimensions(), dimensions);
  std::vector<int> ones = {1,1,1,1};
  ASSERT_EQ(ones, TT.getTTranks());

  TensorTrain_double refTT(dimensions);
  refTT.setUnit(dir);

  EXPECT_NEAR(1., norm2(TT), eps);
  EXPECT_NEAR(1., norm2(refTT), eps);
  EXPECT_NEAR(1., dot(TT, refTT), eps);
}

TEST(PITTS_TensorTrain_fromDense, matrix_2d_4x5_maxRank)
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
  PITTS::MultiVector<double> work, data = toMultiVector(begin(M), end(M), dimensions);
    TensorTrain_double TT = PITTS::fromDense(data, work, dimensions);

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
  PITTS::MultiVector<double> work, data = toMultiVector(begin(M), end(M), dimensions);
    TensorTrain_double TT = PITTS::fromDense(data, work, dimensions, 1.e-16, 3);

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

TEST(PITTS_TensorTrain_fromDense, tensor5d_random_maxRank)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using MultiVector_double = PITTS::MultiVector<double>;

  std::vector<int> shape = {2,3,4,2,3};
  MultiVector_double M(2*3*4*2, 3);
  randomize(M);

  MultiVector_double work;
  TensorTrain_double TT = PITTS::fromDense(M, work, shape, 1.e-16, 2);

  for(auto r: TT.getTTranks())
  {
    ASSERT_LE(r, 2);
  }
}


// anonymous namespace with helper functions
namespace
{
  // check that the distributed (MPI parallel) algorithm obtains the same result as the "serial" algorithm
  void check_mpiGlobal_result(const std::vector<int>& localShape)
  {
    using TensorTrain_double = PITTS::TensorTrain<double>;
    using MultiVector_double = PITTS::MultiVector<double>;
    constexpr auto eps = 1.e-8;

    ASSERT_GE(localShape.size(), 2);
    const auto nDim = localShape.size();

    const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();

    std::vector<int> globalShape = localShape;
    globalShape[0] *= nProcs;
    const long long nLocal = std::accumulate(localShape.begin(), localShape.end(), 1, std::multiplies<long long>());
    const long long nGlobal = nLocal * nProcs;

    // generate random data and distribute it
    using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    mat globalData = mat::Random(nGlobal/globalShape.back(), globalShape.back());
    MPI_Bcast(globalData.data(), nGlobal, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // calculate the global solution on each process
    MultiVector_double Mglobal(nGlobal/globalShape.back(), globalShape.back());
    {
      auto mapMglobal = EigenMap(Mglobal);
      mapMglobal = globalData;
    }
    MultiVector_double work;
    const auto globalTT = PITTS::fromDense(Mglobal, work, globalShape, 1.e-12, 5, false);


    // calculate distributed solution
    MultiVector_double Mlocal(nLocal/localShape.back(), localShape.back());
    {
      auto mapMlocal = EigenMap(Mlocal);
      Eigen::Map<mat> mapGlobalData = Eigen::Map<mat>(globalData.data(), nProcs, nGlobal/nProcs);
      mat localData = mapGlobalData.row(iProc);
      Eigen::Map<mat> mapLocalData = Eigen::Map<mat>(localData.data(), nLocal/localShape.back(), localShape.back());
      mapMlocal = mapLocalData;
    }
    const auto distributedTT = PITTS::fromDense(Mlocal, work, localShape, 1.e-12, 5, true);

    // distributedTT and globalTT should be identical, only the first sub-tensor is distributed onto multiple processes...
    for(int iDim = 1; iDim < nDim; iDim++)
    {
      const auto& subT_ref = globalTT.subTensors()[iDim];
      const auto& subT = distributedTT.subTensors()[iDim];

      ASSERT_EQ(subT_ref.r1(), subT.r1());
      ASSERT_EQ(subT_ref.n(), subT.n());
      ASSERT_EQ(subT_ref.r2(), subT.r2());

      for(int i = 0; i < subT.r1(); i++)
        for(int j = 0; j < subT.n(); j++)
          for(int k = 0; k < subT.r2(); k++)
          {
            // only compare absolute values as the sign of singular vectors is not well defined
            EXPECT_NEAR(std::abs(subT_ref(i,j,k)), std::abs(subT(i,j,k)), eps);
          }
    }

    // first dimension is distributed
    {
      const auto& subT_ref = globalTT.subTensors()[0];
      const auto& subT = distributedTT.subTensors()[0];

      ASSERT_EQ(subT_ref.r1(), subT.r1());
      ASSERT_EQ(globalShape[0], subT_ref.n());
      ASSERT_EQ(localShape[0], subT.n());
      ASSERT_EQ(subT_ref.r2(), subT.r2());

      for(int i = 0; i < subT.r1(); i++)
        for(int k = 0; k < subT.r2(); k++)
          for(int j = 0; j < subT.n(); j++)
          {
            // only compare absolute values as the sign of singular vectors is not well defined
            EXPECT_NEAR(std::abs(subT_ref(i,iProc+j*nProcs,k)), std::abs(subT(i,j,k)), eps);
          }
    }
  }
}

TEST(PITTS_TensorTrain_fromDense, tensor2d_mpiGlobal)
{
  check_mpiGlobal_result({1, 10});
}

TEST(PITTS_TensorTrain_fromDense, another_tensor2d_mpiGlobal)
{
  check_mpiGlobal_result({3, 1});
}

TEST(PITTS_TensorTrain_fromDense, larger_tensor2d_mpiGlobal)
{
  check_mpiGlobal_result({7, 15});
}

TEST(PITTS_TensorTrain_fromDense, tensor3d_mpiGlobal)
{
  check_mpiGlobal_result({1, 5, 5});
}

TEST(PITTS_TensorTrain_fromDense, another_tensor3d_mpiGlobal)
{
  check_mpiGlobal_result({3, 5, 5});
}

TEST(PITTS_TensorTrain_fromDense, tensor5d_mpiGlobal)
{
  check_mpiGlobal_result({1, 5, 4, 5, 3});
}

TEST(PITTS_TensorTrain_fromDense, another_tensor5d_mpiGlobal)
{
  check_mpiGlobal_result({2, 5, 4, 5, 3});
}

TEST(PITTS_TensorTrain_fromDense, boundaryRank_nDim1_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  constexpr auto eps = 1.e-10;

  const int r0 = 3;
  std::vector<int> shape = {5};
  const int rd = 2;
  MultiVector_double M(r0, 5*rd);
  randomize(M);

  MultiVector_double work;
  TensorTrain_double TT = PITTS::fromDense(M, work, shape, 0., -1, false, r0, rd);

  ASSERT_EQ(shape, TT.dimensions());
  const auto& subT = TT.subTensors()[0];
  ASSERT_EQ(r0, subT.r1());
  ASSERT_EQ(rd, subT.r2());

  for(int i = 0; i < subT.r1(); i++)
    for(int j = 0; j < subT.n(); j++)
      for(int k = 0; k < subT.r2(); k++)
      {
        EXPECT_NEAR(M(i,j+k*5), subT(i,j,k), eps);
      }
}

TEST(PITTS_TensorTrain_fromDense, boundaryRank_nDim2_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  const int r0 = 3;
  std::vector<int> shape = {3,5};
  const int rd = 2;
  MultiVector_double M(r0*3, 5*rd);
  randomize(M);

  MultiVector_double work;
  TensorTrain_double TT = PITTS::fromDense(M, work, shape, 0., -1, false, r0, rd);

  ASSERT_EQ(shape, TT.dimensions());
  const auto& subT_l = TT.subTensors().front();
  const auto& subT_r = TT.subTensors().back();
  ASSERT_EQ(r0, subT_l.r1());
  ASSERT_EQ(rd, subT_r.r2());

  Tensor3_double subT = combine(subT_l, subT_r);

  for(int i = 0; i < r0; i++)
    for(int j = 0; j < shape[0]; j++)
      for(int k = 0; k < shape[1]; k++)
        for(int l = 0; l < rd; l++)
        {
          EXPECT_NEAR(M(i+j*r0,k+shape[1]*l), subT(i,j+k*shape[0],l), eps);
        }
}

TEST(PITTS_TensorTrain_fromDense, boundaryRank_nDim5_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  const int r0 = 3;
  const std::vector<int> shape = {4,2,3,4,3};
  const int rd = 2;
  MultiVector_double M(r0*4*2*3*4, 3*rd), refM;
  randomize(M);
  copy(M, refM);

  MultiVector_double work;
  TensorTrain_double TT = PITTS::fromDense(M, work, shape, 0., -1, false, r0, rd);

  ASSERT_EQ(shape, TT.dimensions());
  const auto& subT_l = TT.subTensors().front();
  const auto& subT_r = TT.subTensors().back();
  ASSERT_EQ(r0, subT_l.r1());
  ASSERT_EQ(rd, subT_r.r2());

  std::vector<int> refShape = shape;
  refShape.front() *= r0;
  refShape.back() *= rd;
  TensorTrain_double refTT = PITTS::fromDense(refM, work, refShape, 0., -1, false);

  constexpr auto tensor3_equal = [eps](const Tensor3_double& a, const Tensor3_double& b)
  {
    ASSERT_EQ(a.r1(), b.r1());
    ASSERT_EQ(a.n(), b.n());
    ASSERT_EQ(a.r2(), b.r2());

    for(int i = 0; i < a.r1(); i++)
      for(int j = 0; j < a.n(); j++)
        for(int k = 0; k < a.r2(); k++)
        {
          EXPECT_NEAR(a(i,j,k), b(i,j,k), eps);
        }
  };

  tensor3_equal(refTT.subTensors()[1], TT.subTensors()[1]);
  tensor3_equal(refTT.subTensors()[2], TT.subTensors()[2]);
  tensor3_equal(refTT.subTensors()[3], TT.subTensors()[3]);

  const auto& refSubT_l = refTT.subTensors().front();
  const auto& refSubT_r = refTT.subTensors().back();
  
  ASSERT_EQ(refSubT_l.r1(), 1);
  ASSERT_EQ(refSubT_l.n(), subT_l.r1() * subT_l.n());
  ASSERT_EQ(refSubT_l.r2(), subT_l.r2());
  for(int i = 0; i < subT_l.r1(); i++)
    for(int j = 0; j < subT_l.n(); j++)
      for(int k = 0; k < subT_l.r2(); k++)
      {
        EXPECT_NEAR(refSubT_l(0,i+j*r0,k), subT_l(i,j,k), eps);
      }

  ASSERT_EQ(refSubT_r.r1(), subT_r.r1());
  ASSERT_EQ(refSubT_r.n(), subT_r.n() * subT_r.r2());
  ASSERT_EQ(refSubT_r.r2(), 1);
  for(int i = 0; i < subT_r.r1(); i++)
    for(int j = 0; j < subT_r.n(); j++)
      for(int k = 0; k < subT_r.r2(); k++)
      {
        EXPECT_NEAR(refSubT_r(i,j+k*subT_r.n(),0), subT_r(i,j,k), eps);
      }
}

