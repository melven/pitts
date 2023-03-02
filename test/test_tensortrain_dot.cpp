#include <gtest/gtest.h>
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensor2_random.hpp"
#include "pitts_tensor3_random.hpp"

TEST(PITTS_TensorTrain_dot, rank_1_vector_self)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(1,5);

  TT.setZero();
  EXPECT_EQ(0, dot(TT,TT));

  TT.setOnes();
  EXPECT_NEAR(5., dot(TT,TT), eps);

  TT.setUnit({2});
  EXPECT_NEAR(1, dot(TT,TT), eps);
}


TEST(PITTS_TensorTrain_dot, large_rank_1_vector_self)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(1,50);

  TT.setZero();
  EXPECT_EQ(0, dot(TT,TT));

  TT.setOnes();
  EXPECT_NEAR(50., dot(TT,TT), eps);

  TT.setUnit({2});
  EXPECT_NEAR(1, dot(TT,TT), eps);
}


TEST(PITTS_TensorTrain_dot, rank_1_vector_random)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(1,5), unitTT(1,5);

  randomize(TT);
  double tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    unitTT.setUnit({i});
    tmp += std::pow(dot(TT,unitTT),2);
  }
  EXPECT_NEAR(tmp, dot(TT,TT), eps);
}

TEST(PITTS_TensorTrain_dot, rank_1_vector_random_other)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(1,5), TT2(1,5), unitTT(1,5);

  randomize(TT1);
  randomize(TT2);
  double tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    unitTT.setUnit({i});
    tmp += dot(TT1,unitTT) * dot(TT2,unitTT);
  }
  EXPECT_NEAR(tmp, dot(TT1,TT2), eps);
}

TEST(PITTS_TensorTrain_dot, rank_2_matrix_self)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(2,5);

  TT.setZero();
  EXPECT_EQ(0, dot(TT,TT));

  TT.setOnes();
  EXPECT_NEAR(25., dot(TT,TT), eps);

  TT.setUnit({2,1});
  EXPECT_NEAR(1, dot(TT,TT), eps);
}

TEST(PITTS_TensorTrain_dot, rank_2_matrix_random_self)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(2,5), unitTT(2,5);

  // TT-rank==1
  randomize(TT);
  double tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 5; j++)
    {
      unitTT.setUnit({i,j});
      tmp += std::pow(dot(TT,unitTT),2);
    }
  }
  EXPECT_NEAR(tmp, dot(TT,TT), eps);

  // TT-rank==3
  TT.setTTranks(3);
  randomize(TT);
  tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 5; j++)
    {
      unitTT.setUnit({i,j});
      tmp += std::pow(dot(TT,unitTT),2);
    }
  }
  EXPECT_NEAR(tmp, dot(TT,TT), eps);
}

TEST(PITTS_TensorTrain_dot, rank_2_matrix_random_other)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(2,5), TT2(2,5), unitTT(2,5);

  // TT-rank==1
  randomize(TT1);
  randomize(TT2);
  double tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 5; j++)
    {
      unitTT.setUnit({i,j});
      tmp += dot(TT1,unitTT) * dot(TT2,unitTT);
    }
  }
  EXPECT_NEAR(tmp, dot(TT1,TT2), eps);

  // TT-rank>1
  TT1.setTTranks(3);
  randomize(TT1);
  TT2.setTTranks(2);
  randomize(TT2);
  tmp = 0;
  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 5; j++)
    {
      unitTT.setUnit({i,j});
      tmp += dot(TT1,unitTT) * dot(TT2,unitTT);
    }
  }
  EXPECT_NEAR(tmp, dot(TT1,TT2), eps);
}


TEST(PITTS_TensorTrain_dot, rank_1_vector_unit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(1,3), TT2(1,3);

  TT1.setUnit({0});
  TT2.setUnit({0});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));

  TT2.setUnit({1});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT2.setUnit({2});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT1.setUnit({2});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));
}

TEST(PITTS_TensorTrain_dot, rank_2_matrix_unit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(2,3), TT2(2,3);

  TT1.setUnit({0,0});
  TT2.setUnit({0,0});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));

  TT2.setUnit({1,0});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT2.setUnit({2,0});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT1.setUnit({2,0});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));


  TT1.setUnit({0,1});
  TT2.setUnit({0,1});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));

  TT2.setUnit({1,0});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));
}

TEST(PITTS_TensorTrain_dot, rank_4_tensor_unit)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(4,3), TT2(4,3);

  TT1.setUnit({0,0,1,2});
  TT2.setUnit({0,0,1,2});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));

  TT2.setUnit({1,0,1,2});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT2.setUnit({2,0,1,2});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));

  TT1.setUnit({2,0,1,2});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));


  TT1.setUnit({0,1,1,2});
  TT2.setUnit({0,1,1,2});
  EXPECT_EQ(1, dot(TT1,TT2));
  EXPECT_EQ(1, dot(TT2,TT1));

  TT2.setUnit({1,0,1,2});
  EXPECT_EQ(0, dot(TT1,TT2));
  EXPECT_EQ(0, dot(TT2,TT1));
}

TEST(PITTS_TensorTrain_dot, rank_4_tensor_random_self)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT(4,3), unitTT(4,3);

  // TT-rank==1
  randomize(TT);
  double tmp = 0;
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 3; k++)
        for(int l = 0; l < 3; l++)
        {
          unitTT.setUnit({i,j,k,l});
          tmp += std::pow(dot(TT,unitTT),2);
        }
  EXPECT_NEAR(tmp, dot(TT,TT), eps);

  // TT-rank>1
  TT.setTTranks({2,3,2});
  randomize(TT);
  tmp = 0;
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      for(int k = 0; k < 3; k++)
        for(int l = 0; l < 3; l++)
        {
          unitTT.setUnit({i,j,k,l});
          tmp += std::pow(dot(TT,unitTT),2);
        }
  EXPECT_NEAR(tmp, dot(TT,TT), eps);
}

TEST(PITTS_TensorTrain_dot, large_rank_3_tensor_random_other)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  constexpr auto eps = 1.e-10;

  TensorTrain_double TT1(3,15), TT2(3,15), unitTT(3,15);

  // set larger TT-ranks
  TT1.setTTranks({2,5});
  randomize(TT1);
  TT2.setTTranks({3,2});
  randomize(TT2);
  double tmp = 0;
  for(int i = 0; i < 15; i++)
    for(int j = 0; j < 15; j++)
      for(int k = 0; k < 15; k++)
      {
        unitTT.setUnit({i,j,k});
        tmp += dot(TT1,unitTT) * dot(TT2,unitTT);
      }
  EXPECT_NEAR(tmp, dot(TT1,TT2), eps);
}

TEST(PITTS_TensorTrain_dot, boundary_rank_nDim1)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  // special tensor train where the first dimensions of the first sub tensor is not one
  // (and similarly the last dimension of the last sub tensor)
  
  TensorTrain_double TT1(1,5), TT2(1,5);
  {
    Tensor3_double subT1(2,5,3);
    subT1.setConstant(1);
    Tensor3_double subT2(2,5,3);
    subT2.setConstant(0);
    TT1.setSubTensor(0, std::move(subT1));
    TT2.setSubTensor(0, std::move(subT2));
  }

  EXPECT_NEAR(0, dot(TT1, TT2), eps);
  EXPECT_NEAR(2*5*3, dot(TT1, TT1), eps);

  randomize(TT1);
  randomize(TT2);

  const auto& subT1 = TT1.subTensor(0);
  const auto& subT2 = TT2.subTensor(0);

  double dot_ref = 0;
  for(int i = 0; i < 2; i++)
    for(int j = 0; j < 5; j++)
      for(int k = 0; k < 3; k++)
        dot_ref += subT1(i,j,k) * subT2(i,j,k);

  EXPECT_NEAR(dot_ref, dot(TT1,TT2), eps);
}

TEST(PITTS_TensorTrain_dot, boundary_rank_nDim2)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  // special tensor train where the first dimensions of the first sub tensor is not one
  // (and similarly the last dimension of the last sub tensor)
  TensorTrain_double TT1_ref({3,3,4,4}), TT2_ref({3,3,4,4});
  TensorTrain_double TT1(std::vector<int>{3,4}), TT2(std::vector<int>{3,4});

  TT1_ref.setTTranks(3);
  TT2_ref.setTTranks(3);
  randomize(TT1_ref);
  randomize(TT2_ref);

  // make boundary tensors identity
  {
    Tensor3_double subT1(1,3,3);
    Tensor3_double subT2(1,3,3);
    subT1.setConstant(0);
    subT2.setConstant(0);
    for(int i = 0; i < 3; i++)
    {
      subT1(0,i,i) = 1;
      subT2(0,i,i) = 1;
    }
    TT1_ref.setSubTensor(0, std::move(subT1));
    TT2_ref.setSubTensor(0, std::move(subT2));
  }
  {
    Tensor3_double subT1(3,4,1);
    Tensor3_double subT2(3,4,1);
    subT1.setConstant(0);
    subT2.setConstant(0);
    for(int i = 0; i < 3; i++)
    {
      subT1(i,i,0) = 1;
      subT2(i,i,0) = 1;
    }
    TT1_ref.setSubTensor(3, std::move(subT1));
    TT2_ref.setSubTensor(3, std::move(subT2));
  }

  // copy inner tensors
  TT1.setTTranks(3);
  TT2.setTTranks(3);
  Tensor3_double subT;
  copy(TT1_ref.subTensor(1), subT);
  subT = TT1.setSubTensor(0, std::move(subT));
  copy(TT1_ref.subTensor(2), subT);
  subT = TT1.setSubTensor(1, std::move(subT));
  copy(TT2_ref.subTensor(1), subT);
  subT = TT2.setSubTensor(0, std::move(subT));
  copy(TT2_ref.subTensor(2), subT);
  subT = TT2.setSubTensor(1, std::move(subT));

  const double dot_ref = dot(TT1_ref, TT2_ref);
  EXPECT_NEAR(dot_ref, dot(TT1, TT2), std::abs(dot_ref)*eps);
}

TEST(PITTS_TensorTrain_dot, boundary_rank_nDim6)
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  // special tensor train where the first dimensions of the first sub tensor is not one
  // (and similarly the last dimension of the last sub tensor)
  TensorTrain_double TT1_ref(8,4), TT2_ref(8,4);
  TensorTrain_double TT1(6,4), TT2(6,4);

  TT1_ref.setTTranks(3);
  TT2_ref.setTTranks(3);
  randomize(TT1_ref);
  randomize(TT2_ref);

  // make boundary tensors identity
  {
    Tensor3_double subT1(1,4,3);
    Tensor3_double subT2(1,4,3);
    subT1.setConstant(0);
    subT2.setConstant(0);
    for(int i = 0; i < 3; i++)
    {
      subT1(0,i,i) = 1;
      subT2(0,i,i) = 1;
    }
    TT1_ref.setSubTensor(0, std::move(subT1));
    TT2_ref.setSubTensor(0, std::move(subT2));
  }
  {
    Tensor3_double subT1(3,4,1);
    Tensor3_double subT2(3,4,1);
    subT1.setConstant(0);
    subT2.setConstant(0);
    for(int i = 0; i < 3; i++)
    {
      subT1(i,i,0) = 1;
      subT2(i,i,0) = 1;
    }
    TT1_ref.setSubTensor(7, std::move(subT1));
    TT2_ref.setSubTensor(7, std::move(subT2));
  }

  // copy inner tensors
  TT1.setTTranks(3);
  TT2.setTTranks(3);
  for(int iDim = 0; iDim < 6; iDim++)
  {
    Tensor3_double subT;
    copy(TT1_ref.subTensor(iDim+1), subT);
    subT = TT1.setSubTensor(iDim, std::move(subT));

    copy(TT2_ref.subTensor(iDim+1), subT);
    subT = TT2.setSubTensor(iDim, std::move(subT));
  }

  const double dot_ref = dot(TT1_ref, TT2_ref);
  EXPECT_NEAR(dot_ref, dot(TT1, TT2), std::abs(dot_ref)*eps);
}

TEST(PITTS_TensorTrain_dot, internal_dot_contract1)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

    // contract Tensor3 and Tensor2 along last dimensions: A(:,:,*) * B(:,*)
    // with dimensions n1, n2, n3 and m1, m2=n3
    for(int n1: {5, 35})
      for(int n2: {5, 35})
        for(int n3: {5,})
          for(int m1: {5, 35})
          {
            Tensor3_double A(n1, n2, n3);
            Tensor2_double B(m1, n3);
            Tensor3_double C(3, 3, 3);
            randomize(A);
            randomize(B);
            randomize(C);

            PITTS::internal::dot_contract1(A, B, C);
          
            Tensor3_double C_ref(n1, n2, m1);
            C_ref.setConstant(0);
            for(int i = 0; i < n1; i++)
              for(int j = 0; j < n2; j++)
                for(int k = 0; k < n3; k++)
                  for(int l = 0; l < m1; l++)
                    C_ref(i,j,l) += A(i,j,k) * B(l,k);

            ASSERT_EQ(C_ref.r1(), C.r1());
            ASSERT_EQ(C_ref.n(), C.n());
            ASSERT_EQ(C_ref.r2(), C.r2());
            for(int i = 0; i < n1; i++)
              for(int j = 0; j < n2; j++)
                for(int k = 0; k < m1; k++)
                {
                  ASSERT_NEAR(C_ref(i,j,k), C(i,j,k), eps);
                }
          }
}


TEST(PITTS_TensorTrain_dot, internal_reverse_dot_contract1)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

    // contract Tensor3 and Tensor2 along first dimensions: A(*,:) * B(*,:,:)
    // with dimensions n1, n2, n3 and m1=n1, m2
    for(int n1: {5, 35})
      for(int n2: {5, 35})
        for(int n3: {5,})
          for(int m2: {5, 35})
          {
            Tensor2_double A(n1, m2);
            Tensor3_double B(n1, n2, n3);
            Tensor3_double C(3, 3, 3);
            randomize(A);
            randomize(B);
            randomize(C);

            PITTS::internal::reverse_dot_contract1(A, B, C);
          
            Tensor3_double C_ref(m2, n2, n3);
            C_ref.setConstant(0);
            for(int i = 0; i < n1; i++)
              for(int j = 0; j < n2; j++)
                for(int k = 0; k < n3; k++)
                  for(int l = 0; l < m2; l++)
                    C_ref(l,j,k) += B(i,j,k) * A(i,l);

            ASSERT_EQ(C_ref.r1(), C.r1());
            ASSERT_EQ(C_ref.n(), C.n());
            ASSERT_EQ(C_ref.r2(), C.r2());
            for(int i = 0; i < m2; i++)
              for(int j = 0; j < n2; j++)
                for(int k = 0; k < n3; k++)
                {
                  ASSERT_NEAR(C_ref(i,j,k), C(i,j,k), eps);
                }
          }
}

TEST(PITTS_TensorTrain_dot, internal_dot_contract2)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

    // contract Tensor3 and Tensor3 along the last two dimensions: A(:,*,*) * B(:,*,*)
    // with dimensions n1, n2, n3 and m1, m2=n2, m3=n3
    for(int n1: {5, 35})
      for(int n2: {5, 35})
        for(int n3: {5,})
          for(int m1: {5, 35})
          {
            Tensor3_double A(n1, n2, n3);
            Tensor3_double B(m1, n2, n3);
            Tensor2_double C(3, 3);
            randomize(A);
            randomize(B);
            randomize(C);

            PITTS::internal::dot_contract2(A, B, C);
          
            Tensor2_double C_ref(n1, m1);
            for(int i = 0; i < n1; i++)
              for(int j = 0; j < m1; j++)
                C_ref(i,j) = 0;

            for(int i = 0; i < n1; i++)
              for(int j = 0; j < n2; j++)
                for(int k = 0; k < n3; k++)
                  for(int l = 0; l < m1; l++)
                    C_ref(i,l) += A(i,j,k) * B(l,j,k);

            ASSERT_EQ(C_ref.r1(), C.r1());
            ASSERT_EQ(C_ref.r2(), C.r2());
            for(int i = 0; i < n1; i++)
              for(int j = 0; j < m1; j++)
              {
                ASSERT_NEAR(C_ref(i, j), C(i, j), eps);
              }
          }
}


TEST(PITTS_TensorTrain_dot, internal_reverse_dot_contract2)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

    // contract Tensor3 and Tensor3 along the first two dimensions: A(*,*,:) * B(*,*,:)
    // with dimensions n1, n2, n3 and m1=n1, m2=n2, m3
    for(int n1: {5, 35})
      for(int n2: {5, 35})
        for(int n3: {5,})
          for(int m3: {5, 35})
          {
            std::cout << "test dimensions: " << n1 << " " << n2 << " " << n3 << " " << m3 << "\n";
            Tensor3_double A(n1, n2, n3);
            Tensor3_double B(n1, n2, m3);
            Tensor2_double C(3, 3);
            randomize(A);
            randomize(B);
            randomize(C);

            PITTS::internal::reverse_dot_contract2(A, B, C);
          
            Tensor2_double C_ref(n3, m3);
            for(int i = 0; i < n3; i++)
              for(int j = 0; j < m3; j++)
                C_ref(i,j) = 0;
            
            for(int i = 0; i < n1; i++)
              for(int j = 0; j < n2; j++)
                for(int k = 0; k < n3; k++)
                  for(int l = 0; l < m3; l++)
                    C_ref(k,l) += A(i,j,k) * B(i,j,l);

            ASSERT_EQ(C_ref.r1(), C.r1());
            ASSERT_EQ(C_ref.r2(), C.r2());
            for(int i = 0; i < n3; i++)
              for(int j = 0; j < m3; j++)
              {
                std::cout << "ij: " << i << " " << j << "\n";
                EXPECT_NEAR(C_ref(i, j), C(i, j), eps);
              }
          }
}

TEST(PITTS_TensorTrain_dot, internal_t3_dot)
{
  using Tensor3_double = PITTS::Tensor3<double>;
  constexpr auto eps = 1.e-10;

  // contract Tensor3 and Tensor3 along all dimensions: A(*,*,*) * B(*,*,*)
  for(int n1: {1, 5, 50})
    for(int n2: {1, 5, 50})
      for(int n3: {1, 5, 50})
      {
        Tensor3_double A(n1, n2, n3);
        Tensor3_double B(n1, n2, n3);
        randomize(A);
        randomize(B);

        const double result = PITTS::internal::t3_dot(A, B);

        double result_ref = 0;
        for(int i = 0; i < n1; i++)
          for(int j = 0; j < n2; j++)
            for(int k = 0; k < n3; k++)
              result_ref += A(i,j,k) * B(i,j,k);
        
        EXPECT_NEAR(result_ref, result, eps);
      }
}
