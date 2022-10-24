#include <gtest/gtest.h>
#include "pitts_tensortrain_operator.hpp"
#include <type_traits>

TEST(PITTS_TensorTrainOperator, type_traits)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;

  // implicit copying is not desired
  ASSERT_TRUE(std::is_copy_constructible<TensorTrainOperator_double>());
  ASSERT_FALSE(std::is_copy_assignable<TensorTrainOperator_double>());

  // move / swap is ok
  ASSERT_TRUE(std::is_nothrow_move_constructible<TensorTrainOperator_double>());
  ASSERT_TRUE(std::is_nothrow_move_assignable<TensorTrainOperator_double>());
  ASSERT_TRUE(std::is_nothrow_swappable<TensorTrainOperator_double>());
}

TEST(PITTS_TensorTrainOperator, create_n_m_1)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;

  TensorTrainOperator_double TT1(1,3,4);
  ASSERT_EQ(std::vector<int>({3}), TT1.row_dimensions());
  ASSERT_EQ(std::vector<int>({4}), TT1.column_dimensions());
  ASSERT_EQ(std::vector<int>({}), TT1.getTTranks());
  ASSERT_EQ(1, TT1.tensorTrain().dimensions().size());
  ASSERT_EQ(1, TT1.tensorTrain().subTensor(0).r1());
  ASSERT_EQ(3*4, TT1.tensorTrain().subTensor(0).n());
  ASSERT_EQ(1, TT1.tensorTrain().subTensor(0).r2());
}

TEST(PITTS_TensorTrainOperator, create_n_m_d)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;

  TensorTrainOperator_double TT1(3,7,4);
  ASSERT_EQ(std::vector<int>({7,7,7}), TT1.row_dimensions());
  ASSERT_EQ(std::vector<int>({4,4,4}), TT1.column_dimensions());
  ASSERT_EQ(std::vector<int>({1,1}), TT1.getTTranks());
  ASSERT_EQ(3, TT1.tensorTrain().dimensions().size());
  for(int iDim = 0; iDim < TT1.tensorTrain().dimensions().size(); iDim++)
  {
    const auto& subT = TT1.tensorTrain().subTensor(iDim);
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(7*4, subT.n());
    ASSERT_EQ(1, subT.r2());
  }

  TensorTrainOperator_double TT2(2,3,7,4);
  ASSERT_EQ(std::vector<int>({3,3}), TT2.row_dimensions());
  ASSERT_EQ(std::vector<int>({7,7}), TT2.column_dimensions());
  ASSERT_EQ(std::vector<int>({4}), TT2.getTTranks());
  ASSERT_EQ(2, TT2.tensorTrain().dimensions().size());
  ASSERT_EQ(1, TT2.tensorTrain().subTensor(0).r1());
  ASSERT_EQ(3*7, TT2.tensorTrain().subTensor(0).n());
  ASSERT_EQ(4, TT2.tensorTrain().subTensor(0).r2());
  ASSERT_EQ(4, TT2.tensorTrain().subTensor(1).r1());
  ASSERT_EQ(3*7, TT2.tensorTrain().subTensor(1).n());
  ASSERT_EQ(1, TT2.tensorTrain().subTensor(1).r2());
}

TEST(PITTS_TensorTrainOperator, create_generic)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;

  const std::vector<int> row_dims = {2,3,4,5,6};
  const std::vector<int> col_dims = {6,5,4,3,1};

  TensorTrainOperator_double TT(row_dims, col_dims, 3);
  ASSERT_EQ(row_dims, TT.row_dimensions());
  ASSERT_EQ(col_dims, TT.column_dimensions());
  ASSERT_EQ(std::vector<int>({3,3,3,3}), TT.getTTranks());
  ASSERT_EQ(5, TT.tensorTrain().dimensions().size());
  ASSERT_EQ(1, TT.tensorTrain().subTensor(0).r1());
  ASSERT_EQ(2*6, TT.tensorTrain().subTensor(0).n());
  ASSERT_EQ(3, TT.tensorTrain().subTensor(0).r2());
  ASSERT_EQ(3, TT.tensorTrain().subTensor(1).r1());
  ASSERT_EQ(3*5, TT.tensorTrain().subTensor(1).n());
  ASSERT_EQ(3, TT.tensorTrain().subTensor(1).r2());
  ASSERT_EQ(3, TT.tensorTrain().subTensor(2).r1());
  ASSERT_EQ(4*4, TT.tensorTrain().subTensor(2).n());
  ASSERT_EQ(3, TT.tensorTrain().subTensor(2).r2());
  ASSERT_EQ(3, TT.tensorTrain().subTensor(3).r1());
  ASSERT_EQ(5*3, TT.tensorTrain().subTensor(3).n());
  ASSERT_EQ(3, TT.tensorTrain().subTensor(3).r2());
  ASSERT_EQ(3, TT.tensorTrain().subTensor(4).r1());
  ASSERT_EQ(6*1, TT.tensorTrain().subTensor(4).n());
  ASSERT_EQ(1, TT.tensorTrain().subTensor(4).r2());
}


TEST(PITTS_TensorTrainOperator, setTTranks)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;

  TensorTrainOperator_double TT(3,5,2);
  ASSERT_EQ(std::vector<int>({1,1}), TT.getTTranks());

  TT.setTTranks(2);
  ASSERT_EQ(std::vector<int>({2,2}), TT.getTTranks());
  ASSERT_EQ(3, TT.tensorTrain().dimensions().size());
  ASSERT_EQ(1, TT.tensorTrain().subTensor(0).r1());
  ASSERT_EQ(5*2, TT.tensorTrain().subTensor(0).n());
  ASSERT_EQ(2, TT.tensorTrain().subTensor(0).r2());
  ASSERT_EQ(2, TT.tensorTrain().subTensor(1).r1());
  ASSERT_EQ(5*2, TT.tensorTrain().subTensor(1).n());
  ASSERT_EQ(2, TT.tensorTrain().subTensor(1).r2());
  ASSERT_EQ(2, TT.tensorTrain().subTensor(2).r1());
  ASSERT_EQ(5*2, TT.tensorTrain().subTensor(2).n());
  ASSERT_EQ(1, TT.tensorTrain().subTensor(2).r2());

  TT.setTTranks({3,4});
  ASSERT_EQ(std::vector<int>({3,4}), TT.getTTranks());
  ASSERT_EQ(3, TT.tensorTrain().dimensions().size());
  ASSERT_EQ(1, TT.tensorTrain().subTensor(0).r1());
  ASSERT_EQ(5*2, TT.tensorTrain().subTensor(0).n());
  ASSERT_EQ(3, TT.tensorTrain().subTensor(0).r2());
  ASSERT_EQ(3, TT.tensorTrain().subTensor(1).r1());
  ASSERT_EQ(5*2, TT.tensorTrain().subTensor(1).n());
  ASSERT_EQ(4, TT.tensorTrain().subTensor(1).r2());
  ASSERT_EQ(4, TT.tensorTrain().subTensor(2).r1());
  ASSERT_EQ(5*2, TT.tensorTrain().subTensor(2).n());
  ASSERT_EQ(1, TT.tensorTrain().subTensor(2).r2());
}

TEST(PITTS_TensorTrainOperator, setZero)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;

  TensorTrainOperator_double TT(2,4,3,5);
  TT.setZero();
  // check result is zero everywhere!
  for(int iDim = 0; iDim < TT.tensorTrain().dimensions().size(); iDim++)
  {
    const auto& subT = TT.tensorTrain().subTensor(iDim);
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(4*3, subT.n());
    ASSERT_EQ(1, subT.r2());
    for(int i = 0; i < 4*3; i++)
      ASSERT_EQ(0, subT(0,i,0));
  }
}

TEST(PITTS_TensorTrainOperator, setOnes)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;

  TensorTrainOperator_double TT(2,4,3,5);
  TT.setOnes();
  // check result is one everywhere!
  for(int iDim = 0; iDim < TT.tensorTrain().dimensions().size(); iDim++)
  {
    const auto& subT = TT.tensorTrain().subTensor(iDim);
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(4*3, subT.n());
    ASSERT_EQ(1, subT.r2());
    for(int i = 0; i < 4; i++)
      ASSERT_EQ(1, subT(0,i,0));
  }
}

TEST(PITTS_TensorTrainOperator, setEye)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;

  TensorTrainOperator_double TT(3,4,3,5);
  TT.setEye();
  ASSERT_EQ(std::vector<int>({1,1}), TT.getTTranks());
  // check result is correct everywhere!
  for(int iDim = 0; iDim < TT.tensorTrain().dimensions().size(); iDim++)
  {
    const auto& subT = TT.tensorTrain().subTensor(iDim);
    ASSERT_EQ(1, subT.r1());
    ASSERT_EQ(4*3, subT.n());
    ASSERT_EQ(1, subT.r2());
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 3; j++)
      {
        if( i == j )
        {
          ASSERT_EQ(1, subT(0,TT.index(iDim,i,j),0));
        }
        else
        {
          ASSERT_EQ(0, subT(0,TT.index(iDim,i,j),0));
        }
      }
  }
}

TEST(PITTS_TensorTrainOperator, copy)
{
  using TensorTrainOperator_double = PITTS::TensorTrainOperator<double>;
  using Tensor3_double = PITTS::Tensor3<double>;

  TensorTrainOperator_double TT1(3,4,2,5);
  int dummyVal = 0;
  Tensor3_double newSubT;
  for(int iDim = 0; iDim < TT1.tensorTrain().dimensions().size(); iDim++)
  {
    const auto& subT = TT1.tensorTrain().subTensor(iDim);
    newSubT.resize(subT.r1(), subT.n(), subT.r2());
    for(int i = 0; i < subT.r1(); i++)
      for(int j = 0; j < subT.n(); j++)
        for(int k = 0; k < subT.r2(); k++)
          newSubT(i,j,k) = dummyVal++;
    newSubT = TT1.tensorTrain().setSubTensor(iDim, std::move(newSubT));
  }

  const TensorTrainOperator_double& TT1_const(TT1);
  TensorTrainOperator_double TT2(3,4,2);
  copy(TT1_const, TT2);

  ASSERT_EQ(TT1.tensorTrain().dimensions().size(), TT2.tensorTrain().dimensions().size());
  for(int iDim = 0; iDim < TT1.tensorTrain().dimensions().size(); iDim++)
  {
    const auto& subT1 = TT1.tensorTrain().subTensor(iDim);
    const auto& subT2 = TT2.tensorTrain().subTensor(iDim);
    ASSERT_EQ(subT1.r1(), subT2.r1());
    ASSERT_EQ(subT1.n(), subT2.n());
    ASSERT_EQ(subT1.r2(), subT2.r2());
    for(int i = 0; i < subT1.r1(); i++)
      for(int j = 0; j < subT1.n(); j++)
        for(int k = 0; k < subT1.r2(); k++)
        {
          ASSERT_EQ(subT1(i,j,k), subT2(i,j,k));
        }
  }
}

