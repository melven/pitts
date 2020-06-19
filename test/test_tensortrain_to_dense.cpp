#include <gtest/gtest.h>
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_tensortrain_from_dense.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_dot.hpp"

TEST(PITTS_TensorTrain_toDense, scalar)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::vector<int> dimensions = {1};
  TensorTrain_double TT(dimensions);
  TT.editableSubTensors()[0](0,0,0) = 5.;

  std::array<double,1> scalar;
  PITTS::toDense(TT, begin(scalar), end(scalar));

  ASSERT_NEAR(5., scalar[0], eps);
}

TEST(PITTS_TensorTrain_toDense, dimension_mismatch)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::vector<int> dimensions1 = {10,3};
  TensorTrain_double TT1(dimensions1);
  const std::vector<int> dimensions2 = {2,2,7};
  TensorTrain_double TT2(dimensions2);

  std::array<double,1> scalar;
  std::array<double,10*3> arr30;
  std::array<double,2*2*7> arr28;
  EXPECT_THROW(PITTS::toDense(TT1, begin(scalar), end(scalar)), std::out_of_range);
  EXPECT_NO_THROW(PITTS::toDense(TT1, begin(arr30), end(arr30)));
  EXPECT_THROW(PITTS::toDense(TT2, begin(arr30), end(arr30)), std::out_of_range);
  EXPECT_NO_THROW(PITTS::toDense(TT2, begin(arr28), end(arr28)));
}

TEST(PITTS_TensorTrain_toDense, vector)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::vector<int> dimensions = {5};
  TensorTrain_double TT(dimensions);
  TT.editableSubTensors()[0](0,0,0) = 1.;
  TT.editableSubTensors()[0](0,1,0) = 2.;
  TT.editableSubTensors()[0](0,2,0) = 3.;
  TT.editableSubTensors()[0](0,3,0) = 4.;
  TT.editableSubTensors()[0](0,4,0) = 5.;

  std::array<double,5> data;
  PITTS::toDense(TT, begin(data), end(data));

  ASSERT_NEAR(1., data[0], eps);
  ASSERT_NEAR(2., data[1], eps);
  ASSERT_NEAR(3., data[2], eps);
  ASSERT_NEAR(4., data[3], eps);
  ASSERT_NEAR(5., data[4], eps);
}

TEST(PITTS_TensorTrain_toDense, matrix_2d_5x2_rank1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::vector<int> dimensions = {5,2};
  TensorTrain_double TT(dimensions);
  TT.editableSubTensors()[0](0,0,0) = 1.;
  TT.editableSubTensors()[0](0,1,0) = 2.;
  TT.editableSubTensors()[0](0,2,0) = 3.;
  TT.editableSubTensors()[0](0,3,0) = 4.;
  TT.editableSubTensors()[0](0,4,0) = 5.;
  TT.editableSubTensors()[1](0,0,0) = 1.;
  TT.editableSubTensors()[1](0,1,0) = 2.;

  std::array<double,5*2> data;
  PITTS::toDense(TT, begin(data), end(data));

  ASSERT_NEAR(1., data[0], eps);
  ASSERT_NEAR(2., data[1], eps);
  ASSERT_NEAR(3., data[2], eps);
  ASSERT_NEAR(4., data[3], eps);
  ASSERT_NEAR(5., data[4], eps);
  ASSERT_NEAR(2., data[0+5], eps);
  ASSERT_NEAR(4., data[1+5], eps);
  ASSERT_NEAR(6., data[2+5], eps);
  ASSERT_NEAR(8., data[3+5], eps);
  ASSERT_NEAR(10., data[4+5], eps);
}

TEST(PITTS_TensorTrain_toDense, matrix_2d_2x5_rank1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  const std::vector<int> dimensions = {2,5};
  TensorTrain_double TT(dimensions);
  TT.editableSubTensors()[0](0,0,0) = 1.;
  TT.editableSubTensors()[0](0,1,0) = 2.;
  TT.editableSubTensors()[1](0,0,0) = 1.;
  TT.editableSubTensors()[1](0,1,0) = 2.;
  TT.editableSubTensors()[1](0,2,0) = 3.;
  TT.editableSubTensors()[1](0,3,0) = 4.;
  TT.editableSubTensors()[1](0,4,0) = 5.;

  std::array<double,5*2> data;
  PITTS::toDense(TT, begin(data), end(data));

  ASSERT_NEAR(1., data[0+2*0], eps);
  ASSERT_NEAR(2., data[1+2*0], eps);
  ASSERT_NEAR(2., data[0+2*1], eps);
  ASSERT_NEAR(4., data[1+2*1], eps);
  ASSERT_NEAR(3., data[0+2*2], eps);
  ASSERT_NEAR(6., data[1+2*2], eps);
  ASSERT_NEAR(4., data[0+2*3], eps);
  ASSERT_NEAR(8., data[1+2*3], eps);
  ASSERT_NEAR(5., data[0+2*4], eps);
  ASSERT_NEAR(10., data[1+2*4], eps);
}

TEST(PITTS_TensorTrain_toDense, tensor_3d_rank2_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-8;

  const std::vector<int> dimensions = {2,4,3};
  TensorTrain_double TT(dimensions);
  TT.setTTranks({1,1});
  randomize(TT);

  std::array<double,2*4*3> data;
  PITTS::toDense(TT, begin(data), end(data));

  TensorTrain_double testTT(dimensions);
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 3; k++)
      {
        testTT.setUnit({i,j,k});
        EXPECT_NEAR(dot(testTT,TT), data[i+j*2+k*2*4], eps);
      }
}

TEST(PITTS_TensorTrain_toDense, tensor_5d_from_to_dense_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-8;

  const std::vector<int> dimensions = {2,4,3,5,2};
  Eigen::VectorXd refData = Eigen::VectorXd::Random(2*4*3*5*2);

  // to TT format
  const TensorTrain_double TT = PITTS::fromDense(refData.data(), refData.data()+refData.size(), dimensions);

  // convert back to dense format
  Eigen::VectorXd data(refData.size());
  PITTS::toDense(TT, data.data(), data.data()+data.size());

  for(int i = 0; i < refData.size(); i++)
  {
    EXPECT_NEAR(refData(i), data(i), eps);
  }
}
