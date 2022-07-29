#include <gtest/gtest.h>
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"

namespace
{
  // helper function for copy with a return instead of an argument
  template<typename T>
  auto copy_return(const PITTS::Tensor2<T>& a)
  {
    PITTS::Tensor2<T> b(a.r1(), a.r2());
    copy(a, b);
    return b;
  }
}

TEST(PITTS_Tensor2_EigenAdaptor, simple_const)
{
  using Tensor2_double = PITTS::Tensor2<double>;

  Tensor2_double t2(3,10);

  randomize(t2);
  const Tensor2_double t2_const = copy_return(t2);
  const auto t2Map = ConstEigenMap(t2_const);
  ASSERT_EQ(t2.r1(), t2Map.rows());
  ASSERT_EQ(t2.r2(), t2Map.cols());

  for(int i = 0; i < t2.r1(); i++)
    for(int j = 0; j < t2.r2(); j++)
    {
      EXPECT_EQ(t2(i,j), t2Map(i,j));
    }
}


TEST(PITTS_Tensor2_EigenAdaptor, simple_modify)
{
  using Tensor2_double = PITTS::Tensor2<double>;

  Tensor2_double t2(3,10);

  randomize(t2);
  auto t2Map = EigenMap(t2);
  ASSERT_EQ(t2.r1(), t2Map.rows());
  ASSERT_EQ(t2.r2(), t2Map.cols());

  const Tensor2_double oldT2 = copy_return(t2);
  t2Map = Eigen::MatrixXd::Random(3,10);

  for(int i = 0; i < t2.r1(); i++)
    for(int j = 0; j < t2.r2(); j++)
    {
      EXPECT_EQ(t2(i,j), t2Map(i,j));
      EXPECT_NE(oldT2(i,j), t2(i,j));
    }
}
