#include <gtest/gtest.h>
#include "pitts_tensor2.hpp"
#include <type_traits>

TEST(PITTS_Tensor2, type_traits)
{
  using Tensor2_double = PITTS::Tensor2<double>;

  // implicit copying is not desired
  ASSERT_FALSE(std::is_copy_constructible<Tensor2_double>());
  ASSERT_FALSE(std::is_copy_assignable<Tensor2_double>());

  // move / swap is ok
  ASSERT_TRUE(std::is_nothrow_move_constructible<Tensor2_double>());
  ASSERT_TRUE(std::is_nothrow_move_assignable<Tensor2_double>());
  ASSERT_TRUE(std::is_nothrow_swappable<Tensor2_double>());
}


TEST(PITTS_Tensor2, create)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  Tensor2_double M(3,7);

  ASSERT_EQ(3, M.r1());
  ASSERT_EQ(7, M.r2());
}

TEST(PITTS_Tensor2, resize)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  Tensor2_double M(3,7);

  M.resize(2,3);

  ASSERT_EQ(2, M.r1());
  ASSERT_EQ(3, M.r2());
}

TEST(PITTS_Tensor2, memory_layout_and_zero_padding)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  Tensor2_double M(5,7);
  for(int j = 0; j < 7; j++)
    for(int i = 0; i < 5; i++)
      M(i,j) = 7.;
  
  const auto* Mdata = &M(0,0);
  for(int j = 0; j < 7; j++)
    for(int i = 0; i < 5; i++)
    {
      EXPECT_EQ(7., Mdata[i+j*3]);
    }
  
  for(int k = 7*5; k % PITTS::Chunk<double>::size != 0; k++)
  {
    EXPECT_EQ(0., Mdata[k]);
  }

  M.resize(1,1);
  M(0,0) = 5.;
  ASSERT_EQ(Mdata, &M(0,0));
  EXPECT_EQ(5., Mdata[0]);
  for(int k = 1; k % PITTS::Chunk<double>::size != 0; k++)
  {
    EXPECT_EQ(0., Mdata[k]);
  }
}

TEST(PITTS_Tensor2, operator_indexing)
{
  using Tensor2_double = PITTS::Tensor2<double>;
  Tensor2_double M(3,7);

  // Set to zero
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      M(i,j) = 0;

  // check for zero
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_EQ(0, M(i,j));
    }

  // Set to constant
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      M(i,j) = 77;

  // check
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_EQ(77, M(i,j));
    }

  // set to different values
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
      M(i,j) = i*7+j;

  // check
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 7; j++)
    {
      EXPECT_EQ(i*7+j, M(i,j));
    }
}

/**********************/
/* (Const)Tensor2View */
/**********************/

using namespace PITTS;

TEST(PITTS_Tensor2, View)
{
    const int r1 = 3, r2 = 7;

    Tensor2<double> t2(r1,r2);
    for (int j = 0; j < r2; j++)
      for (int i = 0; i < r1; i++)
        t2(i,j) = i + j*r1; // unique value for each entry
    
    Tensor2View<double> t2view((Chunk<double>*)&t2(0,0), t2.r1(), t2.r2());

    // dimensions must match
    ASSERT_EQ(r1, t2view.r1());
    ASSERT_EQ(r2, t2view.r2());
    // values must match
    for (int j = 0; j < r2; j++)
      for (int i = 0; i < r1; i++)
        EXPECT_EQ(t2(i,j), t2view(i,j));
}

TEST(PITTS_Tensor2, ConstView)
{
    const int r1 = 3, r2 = 7;

    Tensor2<double> t2(r1,r2);
    for (int j = 0; j < r2; j++)
      for (int i = 0; i < r1; i++)
        t2(i,j) = i + j*r1; // unique value for each entry
    
    Tensor2View<double> t2view((Chunk<double>*)&t2(0,0), t2.r1(), t2.r2());

    // dimensions must match
    ASSERT_EQ(r1, t2view.r1());
    ASSERT_EQ(r2, t2view.r2());
    // values must match and changes must propogate
    t2(1,2) = -12;
    t2view(2,1) = -21;
    for (int j = 0; j < r2; j++)
      for (int i = 0; i < r1; i++)
        EXPECT_EQ(t2(i,j), t2view(i,j));
}