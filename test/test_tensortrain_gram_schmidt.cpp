// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_gram_schmidt.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "eigen_test_helper.hpp"

namespace
{
  using TensorTrain_double = PITTS::TensorTrain<double>;
  using arr = Eigen::ArrayXd;
  using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  constexpr auto eps = 2.e-10;
}

TEST(PITTS_TensorTrain_gram_schmidt, single_vector_norm)
{
  TensorTrain_double w(std::vector<int>{3,4,5});
  w.setTTranks({2,3});
  randomize(w);

  TensorTrain_double w_ref(w);

  std::vector<TensorTrain_double> V;
  arr h = PITTS::gramSchmidt(V, w, eps);

  arr h_ref(1);
  h_ref(0) = normalize(w_ref, eps);

  EXPECT_NEAR(h_ref, h, eps);

  ASSERT_EQ(1, V.size());
  const double err = axpby(-1., V[0], 1., w_ref, eps);
  EXPECT_NEAR(0., err, eps);
}

TEST(PITTS_TensorTrain_gram_schmidt, unit_vectors)
{
  std::vector<TensorTrain_double> V;
  mat H = mat::Zero(3*4*5,3*4*5);


  for(int i = 0; i < 3*4*5; i++)
  {
    TensorTrain_double w(std::vector<int>{3,4,5});
    w.setUnit({i%3, (i/3)%4, (i/3)/4});

    const arr h = PITTS::gramSchmidt(V, w, eps, 999, false, " MGS test ", true);
    H.col(i).segment(0, i+1) = h;
  }
  const mat H_ref = mat::Identity(3*4*5, 3*4*5);
  EXPECT_NEAR(H_ref, H, eps);
}

TEST(PITTS_TensorTrain_gram_schmidt, random_vectors)
{
  std::vector<TensorTrain_double> V;
  mat H = mat::Zero(5,5);
  mat X = mat::Zero(7*7*7,5);

  for(int i = 0; i < 5; i++)
  {
    TensorTrain_double w(std::vector<int>{7,7,7});
    w.setTTranks({3,3});
    randomize(w);

    toDense(w, &X(0,i), &X(7*7*7-1,i)+1);
    const arr h = PITTS::gramSchmidt(V, w, eps, 999, false, " MGS test ", true);
    H.col(i).segment(0, i+1) = h;
  }

  ASSERT_EQ(5, V.size());
  mat Q = mat::Zero(7*7*7,5);
  for(int i = 0; i < 5; i++)
  {
    toDense(V[i], &Q(0,i), &Q(7*7*7-1,i)+1);
  }
  mat QtQ = Q.transpose() * Q;
  mat I = mat::Identity(5,5);
  EXPECT_NEAR(I, QtQ, eps);

  mat Q_H = Q * H;
  EXPECT_NEAR(X, Q_H, eps);
}

namespace
{
  // helper function to test different sets of parameters
  void check_gram_schmidt_with_random_vectors(int nIter, bool pivoting, bool modified, bool skipDirs)
  {
    std::vector<TensorTrain_double> V;
    mat H = mat::Zero(5,5);
    mat X = mat::Zero(7*7*7,5);

    for(int i = 0; i < 5; i++)
    {
      TensorTrain_double w(std::vector<int>{7,7,7});
      w.setTTranks({3,3});
      randomize(w);

      toDense(w, &X(0,i), &X(7*7*7-1,i)+1);
      const arr h = PITTS::gramSchmidt(V, w, eps, 999, false, " MGS test ", true, nIter, pivoting, modified, skipDirs);
      H.col(i).segment(0, i+1) = h;
    }

    ASSERT_EQ(5, V.size());
    mat Q = mat::Zero(7*7*7,5);
    for(int i = 0; i < 5; i++)
    {
      toDense(V[i], &Q(0,i), &Q(7*7*7-1,i)+1);
    }
    mat QtQ = Q.transpose() * Q;
    mat I = mat::Identity(5,5);
    EXPECT_NEAR(I, QtQ, eps);

    mat Q_H = Q * H;
    EXPECT_NEAR(X, Q_H, eps);
  }
}

TEST(PITTS_TensorTrain_gram_schmidt, random_vectors_no_pivoting_classical_no_skipDirs)
{
  check_gram_schmidt_with_random_vectors(3, false, false, false);
}

TEST(PITTS_TensorTrain_gram_schmidt, random_vectors_no_pivoting_classical)
{
  check_gram_schmidt_with_random_vectors(3, false, false, true);
}

TEST(PITTS_TensorTrain_gram_schmidt, random_vectors_no_pivoting_modified_no_skipDirs)
{
  check_gram_schmidt_with_random_vectors(2, false, true, false);
}

TEST(PITTS_TensorTrain_gram_schmidt, random_vectors_no_pivoting)
{
  check_gram_schmidt_with_random_vectors(2, false, true, true);
}

TEST(PITTS_TensorTrain_gram_schmidt, random_vectors_pivoting_classical_no_skipDirs)
{
  check_gram_schmidt_with_random_vectors(3, true, false, false);
}

TEST(PITTS_TensorTrain_gram_schmidt, random_vectors_pivoting_classical)
{
  check_gram_schmidt_with_random_vectors(3, true, false, true);
}

TEST(PITTS_TensorTrain_gram_schmidt, random_vectors_pivoting_modified_no_skipDirs)
{
  check_gram_schmidt_with_random_vectors(2, true, true, false);
}

TEST(PITTS_TensorTrain_gram_schmidt, random_vectors_pivoting_modified)
{
  check_gram_schmidt_with_random_vectors(2, true, true, true);
}
