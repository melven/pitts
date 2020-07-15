#include <gtest/gtest.h>
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include <Eigen/Dense>
#include "eigen_test_helper.hpp"


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


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_transformBlock_inplace)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 40;
  constexpr int m = 19;
  constexpr int nChunks = (n-1) / Chunk::size + 1;

  MultiVector X(n,m), X_ref(n,m);
  randomize(X);
  // copy X to X_ref
  for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++)
      X_ref(i,j) = X(i,j);

  PITTS::internal::HouseholderQR::transformBlock(nChunks, m, &X.chunk(0,0), nChunks, &X.chunk(0,0));

  // check that the result is upper triangular
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      if( i > j )
      {
        ASSERT_NEAR(0., X(i,j), eps);
      }
    }
  }

  // use Eigen to check that the singular values and the right singular vectors are identical
  using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::Map<const mat> mapX(&X(0,0), nChunks*Chunk::size, m);
  Eigen::Map<const mat> mapX_ref(&X_ref(0,0), nChunks*Chunk::size, m);
  Eigen::BDCSVD<mat> svd(mapX, Eigen::ComputeThinV);
  Eigen::BDCSVD<mat> svd_ref(mapX_ref, Eigen::ComputeThinV);

  ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  // V can differ by sign, only consider absolute part
  ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_transformBlock_out_of_place)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 40;
  constexpr int m = 19;
  constexpr int nChunks = (n-1) / Chunk::size + 1;

  MultiVector X(2*n,m), X_ref(2*n,m), Xresult(n,m);
  randomize(X);
  randomize(Xresult);
  // copy X to X_ref
  for(int i = 0; i < 2*n; i++)
    for(int j = 0; j < m; j++)
      X_ref(i,j) = X(i,j);

  int lda = (2*n-1) / Chunk::size + 1;

  PITTS::internal::HouseholderQR::transformBlock(nChunks, m, &X.chunk(0,0), lda, &Xresult.chunk(0,0));

  // check that the result is upper triangular
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      if( i > j )
      {
        ASSERT_NEAR(0., Xresult(i,j), eps);
      }
      // X shouldn't change
      ASSERT_EQ(X_ref(i,j), X(i,j));
      ASSERT_EQ(X_ref(i+n,j), X(i+n,j));
    }
  }

  // use Eigen to check that the singular values and the right singular vectors are identical
  using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  Eigen::Map<const mat> mapX(&Xresult(0,0), nChunks*Chunk::size, m);
  Eigen::Map<const mat> mapX_ref(&X_ref(0,0), lda*Chunk::size, m);
  Eigen::BDCSVD<mat> svd(mapX, Eigen::ComputeThinV);
  Eigen::BDCSVD<mat> svd_ref(mapX_ref.topRows(mapX.rows()), Eigen::ComputeThinV);

  ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  // V can differ by sign, only consider absolute part
  ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_copyBlockAndTransformMaybe)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int nChunks = 5;
  constexpr int n = nChunks * Chunk::size;
  constexpr int m = 2;
  MultiVector work(n, m);

  // prepare work array
  int workOffset = 0;
  for(int j = 0; j < m; j++)
    for(int i = 0; i < n; i++)
      work(i,j) = 0;

  constexpr int ldaSrc = 4;
  MultiVector src(ldaSrc*Chunk::size, m);
  randomize(src);

  using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  mat work_ref(0,m);
  Eigen::Map<mat> mapSrc(&src(0,0), ldaSrc*Chunk::size, m);

  for(int iter = 0; iter < 10; iter++)
  {
    const int nSrc = (iter*2) % 5;

    PITTS::internal::HouseholderQR::copyBlockAndTransformMaybe(nSrc, m, &src.chunk(0,0), ldaSrc, nChunks, &work.chunk(0,0), workOffset);

    // check that the result is zero below workOffset
    for(int j = 0; j < m; j++)
    {
      for(int i = workOffset*Chunk::size; i < n; i++)
      {
        ASSERT_NEAR(0., work(i,j), eps);
      }
    }

    // calculate reference result using Eigen
    const int nn = nSrc*Chunk::size;
    work_ref.conservativeResize(work_ref.rows()+nn, Eigen::NoChange);
    work_ref.bottomRows(nn) = mapSrc.topRows(nn);
  }
  // singular values and right singular vectors should match
  Eigen::Map<mat> mapWork(&work(0,0), n, m);
  Eigen::BDCSVD<mat> svd(mapWork, Eigen::ComputeThinV);
  Eigen::BDCSVD<mat> svd_ref(work_ref, Eigen::ComputeThinV);

  ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  // V can differ by sign, only consider absolute part
  ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);
}


namespace
{
  // helper function for testing block_TSQR with different data dimensions, etc
  void test_block_TSQR(int n, int m)
  {
    constexpr auto eps = 1.e-8;
    using Chunk = PITTS::Chunk<double>;
    using MultiVector = PITTS::MultiVector<double>;
    using Tensor2 = PITTS::Tensor2<double>;
    using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    const int nPadded = Chunk::size * ( (n-1) / Chunk::size + 1 );

    MultiVector M(n,m);
    randomize(M);

    // store the original matrix for later
    Eigen::Map<mat> mapM(&M(0,0), nPadded, m);
    mat M_ref = mapM.block(0, 0, n, m);

    Tensor2 R;
    block_TSQR(M, R);
    ASSERT_EQ(m, R.r1());
    ASSERT_EQ(m, R.r2());
    for(int j = 0; j < m; j++)
    {
      for(int i = j+1; i < m; i++)
      {
        ASSERT_NEAR(0., R(i,j), eps);
      }
    }

    // check that the singular values and right singular vectors match...
    Eigen::BDCSVD<mat> svd(PITTS::ConstEigenMap(R), Eigen::ComputeThinV);
    Eigen::BDCSVD<mat> svd_ref(M_ref, Eigen::ComputeThinV);

    ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
    // V can differ by sign, only consider absolute part
    ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);
  }

  // helper function to determine the currently default number of threads in a parallel region
  int get_default_num_threads()
  {
    int numThreads = 1;
#pragma omp parallel
    {
#pragma omp critical (PITTS_TEST_MULTIVECTOR_TSQR)
      numThreads = omp_get_num_threads();
    }
    return numThreads;
  }
}

TEST(PITTS_MultiVector_tsqr, block_TSQR_small_serial)
{
  int nThreads = get_default_num_threads();

  omp_set_num_threads(1);
  test_block_TSQR(50, 10);
  omp_set_num_threads(nThreads);
}


TEST(PITTS_MultiVector_tsqr, block_TSQR_small_4threads)
{
  int nThreads = get_default_num_threads();

  ASSERT_LE(4, omp_get_max_threads());
  omp_set_num_threads(4);

  test_block_TSQR(50, 10);

  omp_set_num_threads(nThreads);
}


TEST(PITTS_MultiVector_tsqr, block_TSQR_large_serial)
{
  int nThreads = get_default_num_threads();

  omp_set_num_threads(1);
  test_block_TSQR(200, 30);
  omp_set_num_threads(nThreads);
}


TEST(PITTS_MultiVector_tsqr, block_TSQR_large_parallel)
{
  test_block_TSQR(200, 30);
}


TEST(PITTS_MultiVector_tsqr, block_TSQR_manyRows_differentNumbersOfThreads)
{
  int nThreads = get_default_num_threads();

  ASSERT_LE(4, omp_get_max_threads());
  for(int iThreads = 1; iThreads < 5; iThreads++)
  {
    omp_set_num_threads(iThreads);
    test_block_TSQR(1000, 1);
  }
  omp_set_num_threads(nThreads);
}
