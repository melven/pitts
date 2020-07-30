#include <gtest/gtest.h>
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include <Eigen/Dense>

namespace
{
  // helper function for copy with a return instead of an argument
  template<typename T>
  auto copy_return(const PITTS::MultiVector<T>& a)
  {
    PITTS::MultiVector<T> b(a.rows(), a.cols());
    for(int j = 0; j < a.cols(); j++)
      for(int i = 0; i < a.rows(); i++)
        b(i,j) = a(i,j);
    return b;
  }
}

TEST(PITTS_MultiVector_EigenAdaptor, simple_const)
{
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double mv(3,10);

  randomize(mv);
  const MultiVector_double mv_const = copy_return(mv);
  const auto mvMap = ConstEigenMap(mv_const);
  ASSERT_EQ(mv.rows(), mvMap.rows());
  ASSERT_EQ(mv.cols(), mvMap.cols());

  for(int i = 0; i < mv.rows(); i++)
    for(int j = 0; j < mv.cols(); j++)
    {
      EXPECT_EQ(mv(i,j), mvMap(i,j));
    }
}


TEST(PITTS_MultiVector_EigenAdaptor, simple_modify)
{
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double mv(3,10);

  randomize(mv);
  auto mvMap = EigenMap(mv);
  ASSERT_EQ(mv.rows(), mvMap.rows());
  ASSERT_EQ(mv.cols(), mvMap.cols());

  const MultiVector_double oldmv = copy_return(mv);
  mvMap = Eigen::MatrixXd::Random(3,10);

  for(int i = 0; i < mv.rows(); i++)
    for(int j = 0; j < mv.cols(); j++)
    {
      EXPECT_EQ(mv(i,j), mvMap(i,j));
      EXPECT_NE(oldmv(i,j), mv(i,j));
    }
}
