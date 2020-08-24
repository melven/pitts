#include <gtest/gtest.h>
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include <Eigen/Dense>
#include "eigen_test_helper.hpp"


TEST(PITTS_MultiVector_transform, invalid_args)
{
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(50,3), Y;
  Tensor2_double M(1,1);

  EXPECT_THROW(transform(X, M, Y), std::invalid_argument);

  M.resize(3,2);

  EXPECT_NO_THROW(transform(X, M, Y));

  EXPECT_THROW(transform(X, M, Y, {10,2}), std::invalid_argument);

  EXPECT_NO_THROW(transform(X, M, Y, {10,10}));
}

TEST(PITTS_MultiVector_transform, single_col_no_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(50,1), Y;
  Tensor2_double M(1,1);

  randomize(X);
  randomize(M);

  transform(X, M, Y);

  const Eigen::MatrixXd Y_ref = ConstEigenMap(X) * ConstEigenMap(M);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, single_col_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(100,1), Y;
  Tensor2_double M(1,1);

  randomize(X);
  randomize(M);

  transform(X, M, Y, {50, 2});

  ASSERT_EQ(50, Y.rows());
  ASSERT_EQ(2, Y.cols());

  const Eigen::MatrixXd Ytmp = ConstEigenMap(X) * ConstEigenMap(M);
  const Eigen::MatrixXd Y_ref = Eigen::Map<const Eigen::MatrixXd>(Ytmp.data(), 50, 2);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, multi_col_no_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(50,3), Y;
  Tensor2_double M(3,2);

  randomize(X);
  randomize(M);

  transform(X, M, Y);

  const Eigen::MatrixXd Y_ref = ConstEigenMap(X) * ConstEigenMap(M);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, multi_col_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(100,3), Y;
  Tensor2_double M(3,2);

  randomize(X);
  randomize(M);

  transform(X, M, Y, {50, 4});

  ASSERT_EQ(50, Y.rows());
  ASSERT_EQ(4, Y.cols());

  const Eigen::MatrixXd Ytmp = ConstEigenMap(X) * ConstEigenMap(M);
  const Eigen::MatrixXd Y_ref = Eigen::Map<const Eigen::MatrixXd>(Ytmp.data(), 50, 4);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, large_no_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(117,4), Y;
  Tensor2_double M(4,5);

  randomize(X);
  randomize(M);

  transform(X, M, Y);

  const Eigen::MatrixXd Y_ref = ConstEigenMap(X) * ConstEigenMap(M);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, large_reshape)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(117,4), Y;
  Tensor2_double M(4,5);

  randomize(X);
  randomize(M);

  transform(X, M, Y, {195, 3});

  ASSERT_EQ(195, Y.rows());
  ASSERT_EQ(3, Y.cols());

  const Eigen::MatrixXd Ytmp = ConstEigenMap(X) * ConstEigenMap(M);
  const Eigen::MatrixXd Y_ref = Eigen::Map<const Eigen::MatrixXd>(Ytmp.data(), 195, 3);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
}

TEST(PITTS_MultiVector_transform, large_reshape2)
{
  const auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;
  using Tensor2_double = PITTS::Tensor2<double>;

  MultiVector_double X(117,4), Y;
  Tensor2_double M(4,5);

  randomize(X);
  randomize(M);

  transform(X, M, Y, {39, 15});

  ASSERT_EQ(39, Y.rows());
  ASSERT_EQ(15, Y.cols());

  const Eigen::MatrixXd Ytmp = ConstEigenMap(X) * ConstEigenMap(M);
  const Eigen::MatrixXd Y_ref = Eigen::Map<const Eigen::MatrixXd>(Ytmp.data(), 39, 15);
  ASSERT_NEAR(Y_ref, ConstEigenMap(Y), eps);
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
  void checkWithRandomData(long long n, long long m, long long k, long long n_, long long m_)
  {
    const auto eps = 1.e-8;

    PITTS::MultiVector<double> X(n,m);
    randomize(X);

    PITTS::Tensor2<double> M(m,k);
    randomize(M);

    PITTS::MultiVector<double> Y;
    prepareOutputWithRubbish(Y, n_, m_);

    transform(X, M, Y, {n_, m_});

    using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    const mat tmp = ConstEigenMap(X) * ConstEigenMap(M);
    ASSERT_EQ(n_*m_, tmp.size());
    Eigen::Map<const mat> Yref(tmp.data(), n_, m_);
    EXPECT_NEAR(Yref, ConstEigenMap(Y), eps);

    // check padding
    for(long long j = 0; j < Y.cols(); j++)
      for(long long i = Y.rows(); i < Y.rowChunks()*PITTS::Chunk<double>::size; i++)
      {
        EXPECT_EQ(0., Y(i,j));
      }
  }

  // output test parameters
  std::string outputParams(const testing::TestParamInfo<std::tuple<int,int,int,int,int>>& info)
  {
    std::string name;
    name += std::to_string(std::get<0>(info.param));
    name += "_" + std::to_string(std::get<1>(info.param));
    name += "_" + std::to_string(std::get<2>(info.param));
    name += "_" + std::to_string(std::get<3>(info.param));
    name += "_" + std::to_string(std::get<4>(info.param));
    return name;
  }
}

class PITTS_MultiVector_transform_param : public testing::TestWithParam<std::tuple<int,int,int,int,int>> {};

TEST_P(PITTS_MultiVector_transform_param, check_result_and_padding)
{
  const auto& [n, m, k, n_, m_] = GetParam();

  checkWithRandomData(n, m, k, n_, m_);
}

INSTANTIATE_TEST_CASE_P(SmallNoReshape, PITTS_MultiVector_transform_param, testing::Values(
    std::make_tuple(1, 1, 1, 1, 1),
    std::make_tuple(1, 5, 1, 1, 1),
    std::make_tuple(5, 1, 1, 5, 1),
    std::make_tuple(5, 3, 1, 5, 1),
    std::make_tuple(5, 7, 4, 5, 4),
    std::make_tuple(7, 3, 4, 7, 4),
    std::make_tuple(3, 5, 7, 3, 7)),
    &outputParams );

INSTANTIATE_TEST_CASE_P(SmallReshape, PITTS_MultiVector_transform_param, testing::Values(
    std::make_tuple(5, 1, 1, 1, 5),
    std::make_tuple(5, 3, 1, 1, 5),
    std::make_tuple(5, 7, 4, 10, 2),
    std::make_tuple(5, 7, 4, 20, 1),
    std::make_tuple(5, 7, 4, 4, 5),
    std::make_tuple(7, 3, 4, 4, 7),
    std::make_tuple(3, 5, 7, 7, 3)),
    &outputParams );

INSTANTIATE_TEST_CASE_P(LargeNoReshape, PITTS_MultiVector_transform_param, testing::Values(
    std::make_tuple(32, 32, 32, 32, 32),
    std::make_tuple(64, 7, 8, 64, 8),
    std::make_tuple(71, 3, 5, 71, 5),
    std::make_tuple(5, 120, 113, 5, 113),
    std::make_tuple(50, 3, 50, 50, 50),
    std::make_tuple(99, 2, 1, 99, 1),
    std::make_tuple(150, 2, 3, 150, 3)),
    &outputParams );

INSTANTIATE_TEST_CASE_P(LargeReshape, PITTS_MultiVector_transform_param, testing::Values(
    std::make_tuple(32, 32, 32, 64, 16),
    std::make_tuple(32, 32, 32, 16, 64),
    std::make_tuple(64, 7, 8, 32, 16),
    std::make_tuple(71, 3, 5, 5, 71),
    std::make_tuple(5, 120, 113, 113, 5),
    std::make_tuple(50, 3, 50, 100, 25),
    std::make_tuple(50, 3, 50, 25, 100),
    std::make_tuple(99, 2, 1, 1, 99),
    std::make_tuple(150, 2, 3, 30, 15)),
    &outputParams );

