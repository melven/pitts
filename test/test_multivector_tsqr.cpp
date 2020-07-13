#include <gtest/gtest.h>
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"
#include <Eigen/Dense>


TEST(PITTS_MultiVector_tsqr, internal_applyHouseholderRotation_inplace)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 40;
  constexpr int m = 5;
  constexpr int nChunks = (n-1) / Chunk::size + 1;

  MultiVector X(n,m), X_ref(n,m);
  randomize(X);
  // copy X to X_ref
  for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++)
      X_ref(i,j) = X(i,j);

  MultiVector v(n,1);
  randomize(v);

  int col = 3;
  int firstRow = 2;

  PITTS::HouseholderQR::internal::applyHouseholderRotation(nChunks, firstRow, col, &v.chunk(0,0), &X.chunk(0,0), nChunks, &X.chunk(0,0));

  // calculate reference result with Eigen
  using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::Map<mat> mapX_ref(&X_ref(0,0), nChunks*Chunk::size, m);
  Eigen::Map<mat> mapV(&v(0,0), nChunks*Chunk::size, 1);

  int offset = firstRow*Chunk::size;
  int n_ = nChunks*Chunk::size - offset;
  mapX_ref.block(offset, col, n_, 1) = (mat::Identity(n_,n_) - mapV.bottomRows(n_) * mapV.bottomRows(n_).transpose()) * mapX_ref.block(offset, col, n_, 1);

  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      ASSERT_NEAR(X_ref(i,j), X(i,j), eps);
    }
  }
}

TEST(PITTS_MultiVector_tsqr, internal_applyHouseholderRotation_out_of_place)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 40;
  constexpr int m = 5;
  constexpr int nChunks = (n-1) / Chunk::size + 1;

  MultiVector X(n,m), X_ref(n,m), X_in(2*n,m);
  randomize(X);
  // copy X to X_ref
  for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++)
      X_ref(i,j) = X(i,j);

  MultiVector v(n,1);
  randomize(v);

  int col = 3;
  int firstRow = 2;
  int lda = (2*n-1) / Chunk::size + 1;

  // set parts of X that will be overwritten to just some number
  for(int i = firstRow; i < nChunks; i++)
    for(int j = 0; j < Chunk::size; j++)
    {
      X_in.chunk(i,col)[j] = X.chunk(i,col)[j];
      X.chunk(i,col)[j] = 77.;
    }

  PITTS::HouseholderQR::internal::applyHouseholderRotation(nChunks, firstRow, col, &v.chunk(0,0), &X_in.chunk(0,0), lda, &X.chunk(0,0));

  // calculate reference result with Eigen
  using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::Map<mat> mapX_ref(&X_ref(0,0), nChunks*Chunk::size, m);
  Eigen::Map<mat> mapV(&v(0,0), nChunks*Chunk::size, 1);

  int offset = firstRow*Chunk::size;
  int n_ = nChunks*Chunk::size - offset;
  mapX_ref.block(offset, col, n_, 1) = (mat::Identity(n_,n_) - mapV.bottomRows(n_) * mapV.bottomRows(n_).transpose()) * mapX_ref.block(offset, col, n_, 1);

  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      ASSERT_NEAR(X_ref(i,j), X(i,j), eps);
    }
  }
}

