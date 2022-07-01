#include <gtest/gtest.h>
#include "pitts_multivector_reshape.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"


TEST(PITTS_MultiVector_reshape, invalid_args)
{
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(50,3), Y;

  EXPECT_THROW(reshape(X, 10, 10, Y), std::invalid_argument);

  EXPECT_NO_THROW(reshape(X, 50, 3, Y));

  EXPECT_THROW(reshape(X, 50, 30, Y), std::invalid_argument);

  EXPECT_NO_THROW(reshape(X, 30, 5, Y));
}

// anonymous namespace
namespace
{
  // helper function that sets all date to an arbitrary value (including padding)
  // used to check that zero padding works fine in all cases
  void prepareOutputWithRubbish(PITTS::MultiVector<double>& M, long long n, long long m)
  {
    // set to´some arbitrary number, then reduce the size to the desired size
    M.resize((n+32)*m,1);
    for(long long i = 0; i < M.rows(); i++)
      M(i,0) = 7.;
  }


  // helper function to check with random data of given size
  void checkWithRandomData(long long n, long long m, long long n_, long long m_)
  {
    const auto eps = 1.e-8;

    PITTS::MultiVector<double> X(n,m);
    randomize(X);

    PITTS::MultiVector<double> Y;
    prepareOutputWithRubbish(Y, n_, m_);

    reshape(X, n_, m_, Y);

    using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    const EigenMatrix tmp = ConstEigenMap(X);
    ASSERT_EQ(n_*m_, tmp.size());
    Eigen::Map<const EigenMatrix> Yref(tmp.data(), n_, m_);
    EXPECT_NEAR(Yref, ConstEigenMap(Y), eps);

    // check padding
    for(long long j = 0; j < Y.cols(); j++)
      for(long long i = Y.rows(); i < Y.rowChunks()*PITTS::Chunk<double>::size; i++)
      {
        EXPECT_EQ(0., Y(i,j));
      }
  }

  // output test parameters
  std::string outputParams(const testing::TestParamInfo<std::tuple<int,int,int,int>>& info)
  {
    std::string name;
    name += std::to_string(std::get<0>(info.param));
    name += "_" + std::to_string(std::get<1>(info.param));
    name += "_" + std::to_string(std::get<2>(info.param));
    name += "_" + std::to_string(std::get<3>(info.param));
    return name;
  }
}

class PITTS_MultiVector_reshape_param : public testing::TestWithParam<std::tuple<int,int,int,int>> {};

TEST_P(PITTS_MultiVector_reshape_param, check_result_and_padding)
{
  const auto& [n, m, n_, m_] = GetParam();

  checkWithRandomData(n, m, n_, m_);
}

INSTANTIATE_TEST_CASE_P(SmallOnlyCopy, PITTS_MultiVector_reshape_param, testing::Values(
    std::make_tuple(1, 1, 1, 1),
    std::make_tuple(1, 5, 1, 5),
    std::make_tuple(5, 1, 5, 1),
    std::make_tuple(5, 3, 5, 3),
    std::make_tuple(5, 7, 5, 7),
    std::make_tuple(7, 3, 7, 3),
    std::make_tuple(3, 5, 3, 5)),
    &outputParams );

INSTANTIATE_TEST_CASE_P(SmallReshape, PITTS_MultiVector_reshape_param, testing::Values(
    std::make_tuple(5, 1, 1, 5),
    std::make_tuple(5, 3, 3, 5),
    std::make_tuple(5, 7, 35, 1),
    std::make_tuple(5, 7, 1, 35),
    std::make_tuple(5, 7, 7, 5)),
    &outputParams );

INSTANTIATE_TEST_CASE_P(LargeOnly2opy, PITTS_MultiVector_reshape_param, testing::Values(
    std::make_tuple(32, 32, 32, 32),
    std::make_tuple(64, 7, 64, 7),
    std::make_tuple(71, 3, 71, 3),
    std::make_tuple(150, 2, 150, 2)),
    &outputParams );

INSTANTIATE_TEST_CASE_P(LargeReshape, PITTS_MultiVector_reshape_param, testing::Values(
    std::make_tuple(32, 32, 64, 16),
    std::make_tuple(32, 32, 16, 64),
    std::make_tuple(64, 7, 32, 14),
    std::make_tuple(71, 5, 5, 71),
    std::make_tuple(5, 113, 113, 5),
    std::make_tuple(50, 50, 100, 25),
    std::make_tuple(50, 50, 25, 100),
    std::make_tuple(99, 1, 1, 99),
    std::make_tuple(150, 3, 30, 15)),
    &outputParams );
