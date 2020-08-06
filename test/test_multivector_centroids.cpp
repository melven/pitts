#include <gtest/gtest.h>
#include "pitts_multivector_centroids.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector.hpp"

TEST(PITTS_MultiVector_centroids, single_vector)
{
  constexpr auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(20,1), Y(20,1);

  randomize(X);
  randomize(Y);

  std::vector<long long> idx(1, 0);
  std::vector<double> w(1, 77.);

  centroids(X, idx, w, Y);

  for(int i = 0; i < 20; i++)
  {
    EXPECT_NEAR(77. * X(i,0), Y(i,0), eps);
  }
}


TEST(PITTS_MultiVector_centroids, n_x_1)
{
  constexpr auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(20,3), Y(20,1);

  randomize(X);
  randomize(Y);

  std::vector<long long> idx = {0, 0, 0};
  std::vector<double> w = {66., 77., 88.};

  centroids(X, idx, w, Y);

  for(int i = 0; i < 20; i++)
  {
    EXPECT_NEAR(66*X(i,0) + 77*X(i,1) + 88*X(i,2), Y(i,0), eps);
  }
}

TEST(PITTS_MultiVector_centroids, n_x_m)
{
  constexpr auto eps = 1.e-8;
  using MultiVector_double = PITTS::MultiVector<double>;

  MultiVector_double X(20,5), Y(20,2);

  randomize(X);
  randomize(Y);

  std::vector<long long> idx = {1, 0, 0, 1, 0};
  std::vector<double> w = {44., 55., 66., 77., 88.};

  centroids(X, idx, w, Y);

  for(int i = 0; i < 20; i++)
  {
    EXPECT_NEAR(55*X(i,1) + 66*X(i,2) + 88*X(i,4), Y(i,0), eps);
    EXPECT_NEAR(44*X(i,0) + 77*X(i,3), Y(i,1), eps);
  }
}
