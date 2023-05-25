#include <gtest/gtest.h>
#include "pitts_tensor2_concat.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"

TEST(PITTS_Tensor2_concat, concat_horizontally)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixX<double>;

  Tensor2_double A(5, 7), B(5, 4), C(5, 11);

  randomize(A);
  randomize(B);
  randomize(C);

  concatLeftRight<double>(A, B, C);

  mat C_ref(5,11);
  C_ref << ConstEigenMap(A), ConstEigenMap(B);

  EXPECT_NEAR(C_ref, ConstEigenMap(C), 0.);
}

TEST(PITTS_Tensor2_concat, concat_horizontally_zero)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixX<double>;

  Tensor2_double A(5, 7), C(5, 11);

  randomize(A);
  randomize(C);

  concatLeftRight<double>(A, std::nullopt, C);

  mat C_ref(5,11);
  C_ref.leftCols(7) = ConstEigenMap(A);
  C_ref.rightCols(4).setZero();
  EXPECT_NEAR(C_ref, ConstEigenMap(C), 0.);

  
  randomize(C);
  concatLeftRight<double>(std::nullopt, A, C);

  C_ref.leftCols(4).setZero();
  C_ref.rightCols(7) = ConstEigenMap(A);
  EXPECT_NEAR(C_ref, ConstEigenMap(C), 0.);


  randomize(C);
  concatLeftRight<double>(std::nullopt, std::nullopt, C);

  C_ref.setZero();
  EXPECT_NEAR(C_ref, ConstEigenMap(C), 0.);
}

TEST(PITTS_Tensor2_concat, concat_vertically)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixX<double>;

  Tensor2_double A(3, 4), B(5, 4), C(8, 4);

  randomize(A);
  randomize(B);
  randomize(C);

  concatTopBottom<double>(A, B, C);

  mat C_ref(8,4);
  C_ref << ConstEigenMap(A), ConstEigenMap(B);
  EXPECT_NEAR(C_ref, ConstEigenMap(C), 0.);
}

TEST(PITTS_Tensor2_concat, concat_vertically_zero)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using mat = Eigen::MatrixX<double>;

  Tensor2_double A(3, 4), C(8, 4);

  randomize(A);
  randomize(C);

  concatTopBottom<double>(A, std::nullopt, C);

  mat C_ref(8,4);
  C_ref.topRows(3) = ConstEigenMap(A);
  C_ref.bottomRows(5).setZero();
  EXPECT_NEAR(C_ref, ConstEigenMap(C), 0.);


  randomize(C);
  concatTopBottom<double>(std::nullopt, A, C);

  C_ref.topRows(5).setZero();
  C_ref.bottomRows(3) = ConstEigenMap(A);
  EXPECT_NEAR(C_ref, ConstEigenMap(C), 0.);


  randomize(C);
  concatTopBottom<double>(std::nullopt, std::nullopt, C);

  C_ref.setZero();
  EXPECT_NEAR(C_ref, ConstEigenMap(C), 0.);
}

TEST(PITTS_Tensor2_concat, concat_horizontally_invalid_dims)
{
  using T2 = PITTS::Tensor2<double>;

  T2 C(5, 3);

  ASSERT_THROW(concatLeftRight<double>(T2(10,1), T2(5,2), C), std::invalid_argument);
  ASSERT_THROW(concatLeftRight<double>(T2(5,1), T2(7,2), C), std::invalid_argument);
  ASSERT_THROW(concatLeftRight<double>(T2(5,2), T2(7,2), C), std::invalid_argument);
  ASSERT_THROW(concatLeftRight<double>(T2(5,4), std::nullopt, C), std::invalid_argument);
  ASSERT_THROW(concatLeftRight<double>(std::nullopt, T2(5,4), C), std::invalid_argument);
}

TEST(PITTS_Tensor2_concat, concat_vertically_invalid_dims)
{
  using T2 = PITTS::Tensor2<double>;

  T2 C(3, 5);

  ASSERT_THROW(concatLeftRight<double>(T2(1,10), T2(2,5), C), std::invalid_argument);
  ASSERT_THROW(concatLeftRight<double>(T2(1,5), T2(2,7), C), std::invalid_argument);
  ASSERT_THROW(concatLeftRight<double>(T2(2,5), T2(2,7), C), std::invalid_argument);
  ASSERT_THROW(concatLeftRight<double>(T2(4,5), std::nullopt, C), std::invalid_argument);
  ASSERT_THROW(concatLeftRight<double>(std::nullopt, T2(4,5), C), std::invalid_argument);
}