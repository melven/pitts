#include <gtest/gtest.h>
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include <Eigen/Dense>
#include "eigen_test_helper.hpp"


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection_inplace)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 100;
  constexpr int m = 5;

  MultiVector X(n,m), X_ref(n,m);
  randomize(X);
  // copy X to X_ref
  for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++)
      X_ref(i,j) = X(i,j);

  MultiVector v(n,1);
  randomize(v);
  int col = 3;
  int firstRow = 1;
  constexpr int nChunks = 2;

  // mak upper part of v zero
  for(int i = (nChunks+firstRow+1)*Chunk::size; i < n; i++)
    v(i,0) = 0;

  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &v.chunk(0,0), &X.chunk(0,0), X.colStrideChunks(), &X.chunk(0,0), X.colStrideChunks()); // memory layout ok because X is small enough

  // calculate reference result with Eigen
  auto mapX_ref = EigenMap(X_ref);
  auto mapV = ConstEigenMap(v);

  int offset = firstRow*Chunk::size;
  int n_ = n - offset;
  mapX_ref.block(offset, col, n_, 1) = (Eigen::MatrixXd::Identity(n_,n_) - mapV.bottomRows(n_) * mapV.bottomRows(n_).transpose()) * mapX_ref.block(offset, col, n_, 1);

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection_inplace_firstRow_bigger_nChunks)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 100;
  constexpr int m = 5;

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
  constexpr int nChunks = 1;

  // mak upper part of v zero
  for(int i = (nChunks+firstRow+1)*Chunk::size; i < n; i++)
    v(i,0) = 0;

  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &v.chunk(0,0), &X.chunk(0,0), X.colStrideChunks(), &X.chunk(0,0), X.colStrideChunks()); // memory layout ok because X is small enough

  // calculate reference result with Eigen
  auto mapX_ref = EigenMap(X_ref);
  auto mapV = ConstEigenMap(v);

  int offset = firstRow*Chunk::size;
  int n_ = n - offset;
  mapX_ref.block(offset, col, n_, 1) = (Eigen::MatrixXd::Identity(n_,n_) - mapV.bottomRows(n_) * mapV.bottomRows(n_).transpose()) * mapX_ref.block(offset, col, n_, 1);

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection_out_of_place)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 100;
  constexpr int m = 5;

  MultiVector X(n,m), X_ref(n,m), X_in(2*n,m);
  randomize(X);
  // copy X to X_ref
  for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++)
      X_ref(i,j) = X(i,j);

  MultiVector v(n,1);
  randomize(v);
  int col = 3;
  int firstRow = 1;
  constexpr int nChunks = 2;

  // mak upper part of v zero
  for(int i = (nChunks+firstRow+1)*Chunk::size; i < n; i++)
    v(i,0) = 0;

  // set parts of X that will be overwritten to just some number
  for(int i = firstRow; i < nChunks; i++)
    for(int j = 0; j < Chunk::size; j++)
    {
      X_in.chunk(i,col)[j] = X.chunk(i,col)[j];
      X.chunk(i,col)[j] = 77.;
    }


  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &v.chunk(0,0), &X_in.chunk(0,0), X_in.colStrideChunks(), &X.chunk(0,0), X.colStrideChunks()); // memory layout ok because X is small enough

  // calculate reference result with Eigen
  auto mapX_ref = EigenMap(X_ref);
  auto mapV = ConstEigenMap(v);

  int offset = firstRow*Chunk::size;
  int n_ = n - offset;
  mapX_ref.block(offset, col, n_, 1) = (Eigen::MatrixXd::Identity(n_,n_) - mapV.bottomRows(n_) * mapV.bottomRows(n_).transpose()) * mapX_ref.block(offset, col, n_, 1);

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection_out_of_place_firstRow_bigger_nChunks)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 100;
  constexpr int m = 5;

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
  constexpr int nChunks = 1;

  // mak upper part of v zero
  for(int i = (nChunks+firstRow+1)*Chunk::size; i < n; i++)
    v(i,0) = 0;

  // X_in is completely ignored as firstRow >= nChunks

  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &v.chunk(0,0), &X_in.chunk(0,0), X_in.colStrideChunks(), &X.chunk(0,0), X.colStrideChunks()); // memory layout ok because X is small enough

  // calculate reference result with Eigen
  auto mapX_ref = EigenMap(X_ref);
  auto mapV = ConstEigenMap(v);

  int offset = firstRow*Chunk::size;
  int n_ = n - offset;
  mapX_ref.block(offset, col, n_, 1) = (Eigen::MatrixXd::Identity(n_,n_) - mapV.bottomRows(n_) * mapV.bottomRows(n_).transpose()) * mapX_ref.block(offset, col, n_, 1);

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection2_inplace)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 100;
  constexpr int m = 5;

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
  int firstRow = 1;
  int nChunks = 2;

  // calculate vTw
  double vTw = 0;
  for(int i = firstRow; i <= nChunks+firstRow; i++)
    for(int j = 0; j < Chunk::size; j++)
      vTw += v.chunk(i,0)[j] * w.chunk(i,0)[j];
  Chunk vTw_chunk;
  for(int i = 0; i < Chunk::size; i++)
    vTw_chunk[i] = vTw;

  PITTS::internal::HouseholderQR::applyReflection2(nChunks, firstRow, col, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X.chunk(0,0), X.colStrideChunks(), &X.chunk(0,0), X.colStrideChunks()); // memory layout ok because X is small enough

  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &w.chunk(0,0), &X_ref.chunk(0,0), X_ref.colStrideChunks(), &X_ref.chunk(0,0), X_ref.colStrideChunks());
  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &v.chunk(0,0), &X_ref.chunk(0,0), X_ref.colStrideChunks(), &X_ref.chunk(0,0), X_ref.colStrideChunks());

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection2_out_of_place)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 100;
  constexpr int m = 5;

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
  int firstRow = 1;
  int nChunks = 2;

  // calculate vTw
  double vTw = 0;
  for(int i = firstRow; i <= nChunks+firstRow; i++)
    for(int j = 0; j < Chunk::size; j++)
      vTw += v.chunk(i,0)[j] * w.chunk(i,0)[j];
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

  PITTS::internal::HouseholderQR::applyReflection2(nChunks, firstRow, col, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X_in.chunk(0,0), X_in.colStrideChunks(), &X.chunk(0,0), X.colStrideChunks()); // memory layout ok because X is small enough

  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &w.chunk(0,0), &X_ref.chunk(0,0), X_ref.colStrideChunks(), &X_ref.chunk(0,0), X_ref.colStrideChunks());
  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &v.chunk(0,0), &X_ref.chunk(0,0), X_ref.colStrideChunks(), &X_ref.chunk(0,0), X_ref.colStrideChunks());

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_transformBlock_inplace)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 16*Chunk::size - 7;   // force padding, we need some extra space in transformBlock
  constexpr int m = 19;
  constexpr int mChunks = (m-1) / Chunk::size + 1;
  constexpr int nTotalChunks = (n-1) / Chunk::size + 1;
  constexpr int nChunks = nTotalChunks - mChunks;

  MultiVector X(n,m), X_ref(n,m);
  randomize(X);
  // make lower triangular part zero...
  for(int j = 0; j < m; j++)
    for(int i = nChunks+j/Chunk::size+1; i < nTotalChunks; i++)
      X.chunk(i,j) = Chunk{};
  // copy X to X_ref
  for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++)
      X_ref(i,j) = X(i,j);

  PITTS::internal::HouseholderQR::transformBlock(nChunks, m, &X.chunk(0,0), X.colStrideChunks(), &X.chunk(0,0), X.colStrideChunks());

  // check that the result is upper triangular
  for(int i = 0; i < mChunks*Chunk::size; i++)
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
  auto mapX = ConstEigenMap(X);
  auto mapX_ref = ConstEigenMap(X_ref);
  std::cout << "X:\n" << mapX << std::endl;
  Eigen::BDCSVD<Eigen::MatrixXd> svd(mapX.topRows(m), Eigen::ComputeThinV);
  Eigen::BDCSVD<Eigen::MatrixXd> svd_ref(mapX_ref, Eigen::ComputeThinV);

  ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  // V can differ by sign, only consider absolute part
  ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_transformBlock_out_of_place)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int n = 16*Chunk::size - 7;   // force padding, we need some extra space in transformBlock
  constexpr int m = 19;
  constexpr int mChunks = (m-1) / Chunk::size + 1;
  constexpr int nTotalChunks = (n-1) / Chunk::size + 1;
  constexpr int nChunks = nTotalChunks - mChunks;

  MultiVector X(n,m), X_ref(n,m), Xresult(n+Chunk::size,m);
  randomize(X);
  randomize(Xresult);
  // make lower triangular part zero...
  for(int j = 0; j < m; j++)
    for(int i = nChunks+j/Chunk::size+1; i < nTotalChunks; i++)
      X.chunk(i,j) = Chunk{};
  // copy X to X_ref
  for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++)
      X_ref(i,j) = X(i,j);
  // prepare lower part of result
  for(int col = 0; col < m; col++)
    for(int i = nChunks; i < nTotalChunks; i++)
      for(int j = 0; j < Chunk::size; j++)
      {
        Xresult.chunk(1+i,col)[j] = X.chunk(i,col)[j];
        X.chunk(i,col)[j] = 77;
      }

  PITTS::internal::HouseholderQR::transformBlock(nChunks, m, &X.chunk(0,0), X.colStrideChunks(), &Xresult.chunk(1,0), Xresult.colStrideChunks());

  // check that the result is upper triangular, copied to the bottom
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      if( i > nChunks*Chunk::size + j )
      {
        EXPECT_NEAR(0., Xresult(Chunk::size+i,j), eps);
      }
      // X shouldn't change
      if( i < nChunks*Chunk::size )
      {
        ASSERT_EQ(X_ref(i,j), X(i,j));
      }
      else
      {
        ASSERT_EQ(77., X(i,j));
      }
    }
  }

  // use Eigen to check that the singular values and the right singular vectors are identical
  auto mapXresult = ConstEigenMap(Xresult);
  std::cout << "X:\n" << mapXresult << std::endl;
  auto mapX_ref = ConstEigenMap(X_ref);
  Eigen::BDCSVD<Eigen::MatrixXd> svd(mapXresult.bottomRows(n-nChunks*Chunk::size), Eigen::ComputeThinV);
  Eigen::BDCSVD<Eigen::MatrixXd> svd_ref(mapX_ref, Eigen::ComputeThinV);

  ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  // V can differ by sign, only consider absolute part
  ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);
}


TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_copyBlockAndTransformReduction)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  constexpr int nChunks = 5;
  constexpr int n = nChunks * Chunk::size;
  constexpr int m = 20;
  constexpr int mChunks = (m-1) / Chunk::size + 1;

  // prepare work array
  constexpr int ldaWork = mChunks+nChunks+1;
  constexpr int nWork = m*ldaWork + 2*nChunks;
  std::unique_ptr<Chunk[]> work(new Chunk[nWork]);
  int workOffset = 0;
  for(int i = 0; i < nWork; i++)
    work[i] = Chunk{};

  MultiVector src(nChunks*Chunk::size, m);
  randomize(src);

  Eigen::MatrixXd work_ref(0,m);
  auto mapSrc = ConstEigenMap(src);

  for(int iter = 0; iter < 10; iter++)
  {
    const int nSrc = (iter*2) % 5 + 1;

    PITTS::internal::HouseholderQR::copyBlockAndTransformReduction(nSrc, m, &src.chunk(0,0), src.colStrideChunks(), nWork, &work[0], ldaWork, workOffset);

    // check that the lower triangular part of the result is zero (stored at workOffset)
    for(int j = 0; j < m; j++)
    {
      for(int i = j+1; i < mChunks*Chunk::size; i++)
      {
        int idx = (workOffset + j*ldaWork)*Chunk::size + i;
        ASSERT_LT(idx, nWork*Chunk::size);
        ASSERT_NEAR(0., work[idx/Chunk::size][idx%Chunk::size], eps);
      }
    }

    // calculate reference result using Eigen
    const int nn = nSrc*Chunk::size;
    work_ref.conservativeResize(work_ref.rows()+nn, Eigen::NoChange);
    work_ref.bottomRows(nn) = mapSrc.topRows(nn);

    std::cout << "iter: " << iter << "\n";

    // singular values and right singular vectors should match
    Eigen::Map<const Eigen::MatrixXd> mapWork(&work[workOffset][0], ldaWork*Chunk::size, m);
    const int mm = std::min(m, (int)work_ref.rows());
    Eigen::BDCSVD<Eigen::MatrixXd> svd(mapWork.topRows(mm), Eigen::ComputeThinV);
    Eigen::BDCSVD<Eigen::MatrixXd> svd_ref(work_ref, Eigen::ComputeThinV);

    ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
    // V can differ by sign, only consider absolute part
    ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);

  }
}


TEST(PITTS_MultiVector_tsqr, internal_combineTwoBlocks)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;

  MultiVector R1;
  MultiVector R2;
  for(int m = 1; m < 77; m+=7)
  {
    // implementation also works with non-triangular factors, so for simplicity just use random square blocks
    R1.resize(m,m);
    R2.resize(m,m);
    randomize(R1);
    randomize(R2);
    // should both be upper triangular
    for(int i = 0; i < m; i++)
      for(int j = i+1; j < m; j++)
        R1(j,i) = R2(j,i) = 0;

    // we need buffers of correctly padded size...
    const auto mChunks = (m-1) / Chunk::size + 1;
    const int totalSize = int(mChunks*m*Chunk::size);
    std::vector<Chunk> buff1(mChunks*m);
    std::vector<Chunk> buff2(mChunks*m);

    for(int j = 0; j < m; j++)
    {
      for(int i = 0; i < mChunks; i++)
      {
        buff1[i+j*mChunks] = R1.chunk(i,j);
        buff2[i+j*mChunks] = R2.chunk(i,j);
      }
    }

    MPI_Datatype mpi_double = MPI_DOUBLE;
    PITTS::internal::HouseholderQR::combineTwoBlocks((const double*)(&(buff1[0][0])), &(buff2[0][0]), &totalSize, &mpi_double);

    // compara singular values with Eigen
    Eigen::MatrixXd R12(2*m,m);
    R12.block(0,0,m,m) = ConstEigenMap(R1);
    R12.block(m,0,m,m) = ConstEigenMap(R2);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_ref(R12);

    Eigen::Map<Eigen::MatrixXd> result(&(buff2[0][0]), mChunks*Chunk::size, m);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(result);

    EXPECT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  }
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

    MultiVector M(n,m);
    randomize(M);

    // store the original matrix for later
    Eigen::MatrixXd M_ref = ConstEigenMap(M);

    Tensor2 R;
    block_TSQR(M, R, 4, false);
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
    Eigen::BDCSVD<Eigen::MatrixXd> svd(PITTS::ConstEigenMap(R), Eigen::ComputeThinV);
    Eigen::BDCSVD<Eigen::MatrixXd> svd_ref(M_ref, Eigen::ComputeThinV);

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


TEST(PITTS_MultiVector_tsqr, block_TSQR_large_varying_sizes_serial)
{
  int nThreads = get_default_num_threads();

  omp_set_num_threads(1);
  for(int m = 1; m < 30; m++)
  {
    std::cout << "Testing #cols = " << m << "\n";
    test_block_TSQR(250, m);
  }
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


TEST(PITTS_MultiVector_tsqr, block_TSQR_mpiGlobal)
{
  constexpr auto eps = 1.e-8;
  using Chunk = PITTS::Chunk<double>;
  using MultiVector = PITTS::MultiVector<double>;
  using Tensor2 = PITTS::Tensor2<double>;

  const long long nTotal = 500;
  const long long m = 4;

  const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();
  const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(nTotal, {iProc,nProcs});
  const long long n = nLast - nFirst + 1;

  Eigen::MatrixXd Mglobal = Eigen::MatrixXd::Random(nTotal, 4);
  MPI_Bcast(Mglobal.data(), nTotal*m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MultiVector M(n,m);
  {
    auto mapM = EigenMap(M);
    mapM = Mglobal.block(nFirst,0,n,m);
  }

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
  Eigen::BDCSVD<Eigen::MatrixXd> svd(ConstEigenMap(R), Eigen::ComputeThinV);
  Eigen::BDCSVD<Eigen::MatrixXd> svd_ref(Mglobal, Eigen::ComputeThinV);

  ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  // V can differ by sign, only consider absolute part
  ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);
}
