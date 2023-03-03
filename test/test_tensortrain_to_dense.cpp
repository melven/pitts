#include <gtest/gtest.h>
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_tensortrain_from_dense_classical.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "eigen_test_helper.hpp"

TEST(PITTS_TensorTrain_toDense, scalar)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  Tensor3_double subT(1,1,1);
  subT(0,0,0) = 5.;
  const std::vector<int> dimensions = {1};
  TensorTrain_double TT(dimensions);
  TT.setSubTensor(0, std::move(subT));

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
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  const std::vector<int> dimensions = {5};
  TensorTrain_double TT(dimensions);
  Tensor3_double subT(1,5,1);
  subT(0,0,0) = 1.;
  subT(0,1,0) = 2.;
  subT(0,2,0) = 3.;
  subT(0,3,0) = 4.;
  subT(0,4,0) = 5.;
  TT.setSubTensor(0, std::move(subT));

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
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  std::vector<Tensor3_double> subTensors(2);
  subTensors[0].resize(1,5,1);
  subTensors[0](0,0,0) = 1.;
  subTensors[0](0,1,0) = 2.;
  subTensors[0](0,2,0) = 3.;
  subTensors[0](0,3,0) = 4.;
  subTensors[0](0,4,0) = 5.;
  subTensors[1].resize(1,2,1);
  subTensors[1](0,0,0) = 1.;
  subTensors[1](0,1,0) = 2.;
  TensorTrain_double TT(std::move(subTensors));

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
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  std::vector<Tensor3_double> subTensors(2);
  subTensors[0].resize(1,2,1);
  subTensors[0](0,0,0) = 1.;
  subTensors[0](0,1,0) = 2.;
  subTensors[1].resize(1,5,1);
  subTensors[1](0,0,0) = 1.;
  subTensors[1](0,1,0) = 2.;
  subTensors[1](0,2,0) = 3.;
  subTensors[1](0,3,0) = 4.;
  subTensors[1](0,4,0) = 5.;
  TensorTrain_double TT(std::move(subTensors));

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
  const Eigen::VectorXd refData = Eigen::VectorXd::Random(2*4*3*5*2);

  // to TT format
  const TensorTrain_double TT = PITTS::fromDense_classical(refData.data(), refData.data()+refData.size(), dimensions);

  // convert back to dense format
  Eigen::VectorXd data(refData.size());
  PITTS::toDense(TT, data.data(), data.data()+data.size());

  EXPECT_NEAR(refData, data, eps);
}

TEST(PITTS_TensorTrain_toDense, boundaryRank_nDim1_ones)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-8;

  Tensor3_double subT(3,7,5);
  subT.resize(3,7,5);
  subT.setConstant(1);
  TensorTrain_double TT(1, 7);
  TT.setSubTensor(0, std::move(subT));

  Eigen::VectorXd refData = Eigen::VectorXd::Ones(3*7*5);

  // convert to dense format
  Eigen::VectorXd data(refData.size());
  PITTS::toDense(TT, data.data(), data.data()+data.size());

  EXPECT_NEAR(refData, data, eps);
}

TEST(PITTS_TensorTrain_toDense, boundaryRank_nDim2_ones)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-8;

  TensorTrain_double TT(2, 7);
  Tensor3_double subT_l(3,7,1);
  Tensor3_double subT_r(1,7,4);
  subT_l.setConstant(1);
  subT_r.setConstant(1);
  TT.setSubTensor(0, std::move(subT_l));
  TT.setSubTensor(1, std::move(subT_r));

  Eigen::VectorXd refData = Eigen::VectorXd::Ones(3*7*7*4);

  // convert to dense format
  Eigen::VectorXd data(refData.size());
  PITTS::toDense(TT, data.data(), data.data()+data.size());

  EXPECT_NEAR(refData, data, eps);
}

TEST(PITTS_TensorTrain_toDense, boundaryRank_nDim5_ones)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-8;

  TensorTrain_double TT({3,4,2,2,3});
  TT.setOnes();
  Tensor3_double subT_l(5,3,1);
  Tensor3_double subT_r(1,3,2);
  subT_l.setConstant(1);
  subT_r.setConstant(1);
  TT.setSubTensor(0, std::move(subT_l));
  TT.setSubTensor(4, std::move(subT_r));

  Eigen::VectorXd refData = Eigen::VectorXd::Ones(5 * 3*4*2*2*3 * 2);

  // convert to dense format
  Eigen::VectorXd data(refData.size());
  PITTS::toDense(TT, data.data(), data.data()+data.size());

  EXPECT_NEAR(refData, data, eps);
}

TEST(PITTS_TensorTrain_toDense, boundaryRank_nDim1_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-8;

  TensorTrain_double TT(1, 7);
  {
    Tensor3_double subT(3,7,5);
    randomize(subT);
    TT.setSubTensor(0, std::move(subT));
  }

  TensorTrain_double refTT(std::vector<int>{3,7,5});
  refTT.setTTranks({3,5});
  {
    Tensor3_double subT;
    copy(TT.subTensor(0), subT);
    refTT.setSubTensor(1, std::move(subT));
    Tensor3_double subT_l(1,3,3);
    Tensor3_double subT_r(5,5,1);
    for(int i = 0; i < 3; i++)
      for(int j = 0; j < 3; j++)
        subT_l(0,i,j) = i == j ? 1 : 0;
    for(int i = 0; i < 5; i++)
      for(int j = 0; j < 5; j++)
        subT_r(i,j,0) = i == j ? 1 : 0;
    refTT.setSubTensor(0, std::move(subT_l));
    refTT.setSubTensor(2, std::move(subT_r));
  }

  Eigen::VectorXd refData(3*7*5);
  PITTS::toDense(refTT, refData.data(), refData.data()+refData.size());

  // convert to dense format
  Eigen::VectorXd data(refData.size());
  PITTS::toDense(TT, data.data(), data.data()+data.size());

  EXPECT_NEAR(refData, data, eps);
}

TEST(PITTS_TensorTrain_toDense, boundaryRank_nDim2_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-8;

  TensorTrain_double TT(2, 7);
  {
    Tensor3_double subT_l(3,7,1);
    Tensor3_double subT_r(1,7,4);
    randomize(subT_l);
    randomize(subT_r);
    TT.setSubTensor(0, std::move(subT_l));
    TT.setSubTensor(1, std::move(subT_r));
  }

  TensorTrain_double refTT({3,7,7,4});
  refTT.setTTranks({3,1,4});
  {
    Tensor3_double subT;
    copy(TT.subTensor(0), subT);
    subT = refTT.setSubTensor(1, std::move(subT));
    copy(TT.subTensor(1), subT);
    subT = refTT.setSubTensor(2, std::move(subT));
    Tensor3_double subT_l(1,3,3);
    Tensor3_double subT_r(4,4,1);
    for(int i = 0; i < 3; i++)
      for(int j = 0; j < 3; j++)
        subT_l(0,i,j) = i == j ? 1 : 0;
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 4; j++)
        subT_r(i,j,0) = i == j ? 1 : 0;
    refTT.setSubTensor(0, std::move(subT_l));
    refTT.setSubTensor(3, std::move(subT_r));
  }

  Eigen::VectorXd refData(3*7*7*4);
  PITTS::toDense(refTT, refData.data(), refData.data()+refData.size());

  // convert to dense format
  Eigen::VectorXd data(refData.size());
  PITTS::toDense(TT, data.data(), data.data()+data.size());

  EXPECT_NEAR(refData, data, eps);
}

TEST(PITTS_TensorTrain_toDense, boundaryRank_nDim5_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-8;

  TensorTrain_double TT({3,4,2,2,3});
  randomize(TT);
  {
    Tensor3_double subT_l(5,3,1);
    Tensor3_double subT_r(1,3,2);
    randomize(subT_l);
    randomize(subT_r);
    TT.setSubTensor(0, std::move(subT_l));
    TT.setSubTensor(4, std::move(subT_r));
  }


  TensorTrain_double refTT({5, 3,4,2,2,3, 2});
  refTT.setTTranks({5,1,1,1,1,2});
  {
    Tensor3_double subT;
    copy(TT.subTensor(0), subT);
    subT = refTT.setSubTensor(1, std::move(subT));
    copy(TT.subTensor(1), subT);
    subT = refTT.setSubTensor(2, std::move(subT));
    copy(TT.subTensor(2), subT);
    subT = refTT.setSubTensor(3, std::move(subT));
    copy(TT.subTensor(3), subT);
    subT = refTT.setSubTensor(4, std::move(subT));
    copy(TT.subTensor(4), subT);
    subT = refTT.setSubTensor(5, std::move(subT));
    Tensor3_double subT_l(1,5,5);
    Tensor3_double subT_r(2,2,1);
    for(int i = 0; i < 5; i++)
      for(int j = 0; j < 5; j++)
        subT_l(0,i,j) = i == j ? 1 : 0;
    for(int i = 0; i < 2; i++)
      for(int j = 0; j < 2; j++)
        subT_r(i,j,0) = i == j ? 1 : 0;
    refTT.setSubTensor(0, std::move(subT_l));
    refTT.setSubTensor(6, std::move(subT_r));
  }

  Eigen::VectorXd refData(5 * 3*4*2*2*3 * 2);
  PITTS::toDense(refTT, refData.data(), refData.data()+refData.size());

  // convert to dense format
  Eigen::VectorXd data(refData.size());
  PITTS::toDense(TT, data.data(), data.data()+data.size());

  EXPECT_NEAR(refData, data, eps);
}
