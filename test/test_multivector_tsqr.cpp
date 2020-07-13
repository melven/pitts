#include <gtest/gtest.h>
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"
#include <Eigen/Dense>


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyRotation_inplace)
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

  PITTS::internal::HouseholderQR::applyRotation(nChunks, firstRow, col, &v.chunk(0,0), &X.chunk(0,0), nChunks, &X.chunk(0,0));

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


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyRotation_out_of_place)
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

  PITTS::internal::HouseholderQR::applyRotation(nChunks, firstRow, col, &v.chunk(0,0), &X_in.chunk(0,0), lda, &X.chunk(0,0));

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


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyRotation2_inplace)
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
  MultiVector w(n,1);
  randomize(w);

  int col = 3;
  int firstRow = 2;

  // calculate vTw
  double vTw = 0;
  for(int i = firstRow*Chunk::size; i < n; i++)
      vTw += v(i,0) * w(i,0);
  Chunk vTw_chunk;
  for(int i = 0; i < Chunk::size; i++)
    vTw_chunk[i] = vTw;

  PITTS::internal::HouseholderQR::applyRotation2(nChunks, firstRow, col, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X.chunk(0,0), nChunks, &X.chunk(0,0));

  PITTS::internal::HouseholderQR::applyRotation(nChunks, firstRow, col, &w.chunk(0,0), &X_ref.chunk(0,0), nChunks, &X_ref.chunk(0,0));
  PITTS::internal::HouseholderQR::applyRotation(nChunks, firstRow, col, &v.chunk(0,0), &X_ref.chunk(0,0), nChunks, &X_ref.chunk(0,0));

  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      ASSERT_NEAR(X_ref(i,j), X(i,j), eps);
    }
  }
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyRotation2_out_of_place)
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
  MultiVector w(n,1);
  randomize(w);

  int col = 3;
  int firstRow = 2;
  int lda = (2*n-1) / Chunk::size + 1;

  // calculate vTw
  double vTw = 0;
  for(int i = firstRow*Chunk::size; i < n; i++)
      vTw += v(i,0) * w(i,0);
  Chunk vTw_chunk;
  for(int i = 0; i < Chunk::size; i++)
    vTw_chunk[i] = vTw;


  // set parts of X that will be overwritten to just some number
  for(int i = firstRow; i < nChunks; i++)
    for(int j = 0; j < Chunk::size; j++)
    {
      X_in.chunk(i,col)[j] = X.chunk(i,col)[j];
      X.chunk(i,col)[j] = 77.;
    }

  PITTS::internal::HouseholderQR::applyRotation2(nChunks, firstRow, col, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X_in.chunk(0,0), lda, &X.chunk(0,0));

  PITTS::internal::HouseholderQR::applyRotation(nChunks, firstRow, col, &w.chunk(0,0), &X_ref.chunk(0,0), nChunks, &X_ref.chunk(0,0));
  PITTS::internal::HouseholderQR::applyRotation(nChunks, firstRow, col, &v.chunk(0,0), &X_ref.chunk(0,0), nChunks, &X_ref.chunk(0,0));

  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      ASSERT_NEAR(X_ref(i,j), X(i,j), eps);
    }
  }
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyRotation2x2_inplace)
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
  MultiVector w(n,1);
  randomize(w);

  int col = 3;
  int firstRow = 2;

  // calculate vTw
  double vTw = 0;
  for(int i = firstRow*Chunk::size; i < n; i++)
      vTw += v(i,0) * w(i,0);
  Chunk vTw_chunk;
  for(int i = 0; i < Chunk::size; i++)
    vTw_chunk[i] = vTw;

  PITTS::internal::HouseholderQR::applyRotation2x2(nChunks, firstRow, col, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X.chunk(0,0), nChunks, &X.chunk(0,0));

  PITTS::internal::HouseholderQR::applyRotation2(nChunks, firstRow, col, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X_ref.chunk(0,0), nChunks, &X_ref.chunk(0,0));
  PITTS::internal::HouseholderQR::applyRotation2(nChunks, firstRow, col+1, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X_ref.chunk(0,0), nChunks, &X_ref.chunk(0,0));

  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      ASSERT_NEAR(X_ref(i,j), X(i,j), eps);
    }
  }
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyRotation2x2_out_of_place)
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
  MultiVector w(n,1);
  randomize(w);

  int col = 3;
  int firstRow = 2;
  int lda = (2*n-1) / Chunk::size + 1;

  // calculate vTw
  double vTw = 0;
  for(int i = firstRow*Chunk::size; i < n; i++)
      vTw += v(i,0) * w(i,0);
  Chunk vTw_chunk;
  for(int i = 0; i < Chunk::size; i++)
    vTw_chunk[i] = vTw;


  // set parts of X that will be overwritten to just some number
  for(int i = firstRow; i < nChunks; i++)
    for(int j = 0; j < Chunk::size; j++)
    {
      X_in.chunk(i,col)[j] = X.chunk(i,col)[j];
      X_in.chunk(i,col+1)[j] = X.chunk(i,col+1)[j];
      X.chunk(i,col)[j] = 77.;
      X.chunk(i,col+1)[j] = 55.;
    }

  PITTS::internal::HouseholderQR::applyRotation2x2(nChunks, firstRow, col, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X_in.chunk(0,0), lda, &X.chunk(0,0));

  PITTS::internal::HouseholderQR::applyRotation2(nChunks, firstRow, col, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X_ref.chunk(0,0), nChunks, &X_ref.chunk(0,0));
  PITTS::internal::HouseholderQR::applyRotation2(nChunks, firstRow, col+1, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X_ref.chunk(0,0), nChunks, &X_ref.chunk(0,0));

  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      ASSERT_NEAR(X_ref(i,j), X(i,j), eps);
    }
  }
}
