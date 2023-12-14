// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_multivector_tsqr_impl.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_parallel.hpp"
#include "eigen_test_helper.hpp"
#include <complex>
#include <limits>
#include <cmath>

template<typename T>
class PITTS_MultiVector_tsqr: public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_CASE(PITTS_MultiVector_tsqr, TestTypes);

TYPED_TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection_inplace)
{
  using Type = TestFixture::Type;
  using RealType = decltype(std::real(Type()));
  const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
  using Chunk = PITTS::Chunk<Type>;
  using MultiVector = PITTS::MultiVector<Type>;

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
  mapX_ref.block(offset, col, n_, 1) = (Eigen::MatrixX<Type>::Identity(n_,n_) - mapV.bottomRows(n_) * mapV.bottomRows(n_).adjoint()) * mapX_ref.block(offset, col, n_, 1);

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TYPED_TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection_inplace_firstRow_bigger_nChunks)
{
  using Type = TestFixture::Type;
  using RealType = decltype(std::real(Type()));
  const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
  using Chunk = PITTS::Chunk<Type>;
  using MultiVector = PITTS::MultiVector<Type>;

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
  mapX_ref.block(offset, col, n_, 1) = (Eigen::MatrixX<Type>::Identity(n_,n_) - mapV.bottomRows(n_) * mapV.bottomRows(n_).adjoint()) * mapX_ref.block(offset, col, n_, 1);

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TYPED_TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection_out_of_place)
{
  using Type = TestFixture::Type;
  using RealType = decltype(std::real(Type()));
  const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
  using Chunk = PITTS::Chunk<Type>;
  using MultiVector = PITTS::MultiVector<Type>;

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
  mapX_ref.block(offset, col, n_, 1) = (Eigen::MatrixX<Type>::Identity(n_,n_) - mapV.bottomRows(n_) * mapV.bottomRows(n_).adjoint()) * mapX_ref.block(offset, col, n_, 1);

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TYPED_TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection_out_of_place_firstRow_bigger_nChunks)
{
  using Type = TestFixture::Type;
  using RealType = decltype(std::real(Type()));
  const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
  using Chunk = PITTS::Chunk<Type>;
  using MultiVector = PITTS::MultiVector<Type>;

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
  mapX_ref.block(offset, col, n_, 1) = (Eigen::MatrixX<Type>::Identity(n_,n_) - mapV.bottomRows(n_) * mapV.bottomRows(n_).adjoint()) * mapX_ref.block(offset, col, n_, 1);

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TYPED_TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection2_inplace)
{
  using Type = TestFixture::Type;
  using RealType = decltype(std::real(Type()));
  const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
  using Chunk = PITTS::Chunk<Type>;
  using MultiVector = PITTS::MultiVector<Type>;

  constexpr auto conj = [](Type x) -> Type
  {
    if constexpr (std::is_same_v<Type, RealType>)
      return x;
    else
      return std::conj(x);
  };

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
  Type vTw = 0;
  for(int i = firstRow; i <= nChunks+firstRow; i++)
    for(int j = 0; j < Chunk::size; j++)
      vTw += conj(v.chunk(i,0)[j]) * w.chunk(i,0)[j];
  Chunk vTw_chunk;
  for(int i = 0; i < Chunk::size; i++)
    vTw_chunk[i] = vTw;

  PITTS::internal::HouseholderQR::applyReflection2(nChunks, firstRow, col, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X.chunk(0,0), X.colStrideChunks(), &X.chunk(0,0), X.colStrideChunks(), false); // memory layout ok because X is small enough

  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &w.chunk(0,0), &X_ref.chunk(0,0), X_ref.colStrideChunks(), &X_ref.chunk(0,0), X_ref.colStrideChunks());
  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &v.chunk(0,0), &X_ref.chunk(0,0), X_ref.colStrideChunks(), &X_ref.chunk(0,0), X_ref.colStrideChunks());

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TYPED_TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_applyReflection2_out_of_place)
{
  using Type = TestFixture::Type;
  using RealType = decltype(std::real(Type()));
  const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
  using Chunk = PITTS::Chunk<Type>;
  using MultiVector = PITTS::MultiVector<Type>;

  constexpr auto conj = [](Type x) -> Type
  {
    if constexpr (std::is_same_v<Type, RealType>)
      return x;
    else
      return std::conj(x);
  };

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
  Type vTw = 0;
  for(int i = firstRow; i <= nChunks+firstRow; i++)
    for(int j = 0; j < Chunk::size; j++)
      vTw += conj(v.chunk(i,0)[j]) * w.chunk(i,0)[j];
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

  PITTS::internal::HouseholderQR::applyReflection2(nChunks, firstRow, col, &w.chunk(0,0), &v.chunk(0,0), vTw_chunk, &X_in.chunk(0,0), X_in.colStrideChunks(), &X.chunk(0,0), X.colStrideChunks(), false); // memory layout ok because X is small enough

  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &w.chunk(0,0), &X_ref.chunk(0,0), X_ref.colStrideChunks(), &X_ref.chunk(0,0), X_ref.colStrideChunks());
  PITTS::internal::HouseholderQR::applyReflection(nChunks, firstRow, col, &v.chunk(0,0), &X_ref.chunk(0,0), X_ref.colStrideChunks(), &X_ref.chunk(0,0), X_ref.colStrideChunks());

  ASSERT_NEAR(ConstEigenMap(X_ref), ConstEigenMap(X), eps);
}


TYPED_TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_transformBlock_inplace)
{
  using Type = TestFixture::Type;
  using RealType = decltype(std::real(Type()));
  const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
  using Chunk = PITTS::Chunk<Type>;
  using MultiVector = PITTS::MultiVector<Type>;

  constexpr int n = 15*Chunk::size + 1;   // force padding, we need some extra space in transformBlock
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

  PITTS::internal::HouseholderQR::transformBlock(nChunks, m, &X.chunk(0,0), X.colStrideChunks(), &X.chunk(0,0), X.colStrideChunks(), nChunks, false);

  // check that the result is upper triangular
  for(int i = 0; i < mChunks*Chunk::size; i++)
  {
    for(int j = 0; j < m; j++)
    {
      if( i > j )
      {
        ASSERT_NEAR(RealType(0), std::real(X(i,j)), eps);
        ASSERT_NEAR(RealType(0), std::imag(X(i,j)), eps);
      }
    }
  }

  // use Eigen to check that the singular values and the right singular vectors are identical
  auto mapX = ConstEigenMap(X);
  auto mapX_ref = ConstEigenMap(X_ref);
  //std::cout << "X:\n" << mapX << std::endl;
#if EIGEN_VERSION_AT_LEAST(3,4,90)
  Eigen::BDCSVD<Eigen::MatrixX<Type>, Eigen::ComputeThinV> svd(mapX.topRows(m));
  Eigen::BDCSVD<Eigen::MatrixX<Type>, Eigen::ComputeThinV> svd_ref(mapX_ref);
#else
  Eigen::BDCSVD<Eigen::MatrixX<Type>> svd(mapX.topRows(m), Eigen::ComputeThinV);
  Eigen::BDCSVD<Eigen::MatrixX<Type>> svd_ref(mapX_ref, Eigen::ComputeThinV);
#endif

  ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  // V can differ by sign, only consider absolute part
  ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);
}


TYPED_TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_transformBlock_inplace_twoTriangularBlocks)
{
  using Type = TestFixture::Type;
  using RealType = decltype(std::real(Type()));
  const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
  using Chunk = PITTS::Chunk<Type>;
  using MultiVector = PITTS::MultiVector<Type>;

  constexpr int m = 38;
  constexpr int mChunks = (m-1) / Chunk::size + 1;
  constexpr int nTotalChunks = 2*mChunks+2;
  constexpr int nChunks = mChunks;
  constexpr int n = mChunks * Chunk::size;
  constexpr int nTotal = nTotalChunks * Chunk::size;

  MultiVector X(nTotal,m), X_ref(nTotal,m);
  randomize(X);
  // make lower triangular part zero...
  for(int j = 0; j < m; j++)
    for(int i = j+1; i < n; i++)
      X(i,j) = X(i+n,j) = 0;
  for(int j = 0; j < m; j++)
    for(int i = 2*n; i < nTotal; i++)
      X(i,j) = 0;
  // copy X to X_ref
  copy(X, X_ref);

  PITTS::internal::HouseholderQR::transformBlock(nChunks, m, &X.chunk(0,0), X.colStrideChunks(), &X.chunk(0,0), X.colStrideChunks(), nChunks, true);

  // check that the result is upper triangular
  for(int i = 0; i < mChunks*Chunk::size; i++)
  {
    for(int j = 0; j < m; j++)
    {
      if( i > j )
      {
        ASSERT_NEAR(RealType(0), std::real(X(i,j)), eps);
        ASSERT_NEAR(RealType(0), std::imag(X(i,j)), eps);
      }
    }
  }

  // use Eigen to check that the singular values and the right singular vectors are identical
  auto mapX = ConstEigenMap(X);
  auto mapX_ref = ConstEigenMap(X_ref);
  //std::cout << "X:\n" << mapX << std::endl;
#if EIGEN_VERSION_AT_LEAST(3,4,90)
  Eigen::BDCSVD<Eigen::MatrixX<Type>, Eigen::ComputeThinV> svd(mapX.topRows(m));
  Eigen::BDCSVD<Eigen::MatrixX<Type>, Eigen::ComputeThinV> svd_ref(mapX_ref);
#else
  Eigen::BDCSVD<Eigen::MatrixX<Type>> svd(mapX.topRows(m), Eigen::ComputeThinV);
  Eigen::BDCSVD<Eigen::MatrixX<Type>> svd_ref(mapX_ref, Eigen::ComputeThinV);
#endif

  ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  // V can differ by sign, only consider absolute part
  ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);
}


TYPED_TEST(PITTS_MultiVector_tsqr, internal_HouseholderQR_transformBlock_out_of_place)
{
  using Type = TestFixture::Type;
  using RealType = decltype(std::real(Type()));
  const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
  using Chunk = PITTS::Chunk<Type>;
  using MultiVector = PITTS::MultiVector<Type>;

  constexpr int n = 16*Chunk::size - 3;   // force padding, we need some extra space in transformBlock
  constexpr int m = 9;
  constexpr int mChunks = (m-1) / Chunk::size + 1;
  constexpr int nTotalChunks = (n-1) / Chunk::size + 1;
  constexpr int nChunks = nTotalChunks - mChunks;

  MultiVector X(n,m), X_ref(n,m), Xresult(n+2*Chunk::size,m);
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
        Xresult.chunk(2+i,col)[j] = X.chunk(i,col)[j];
        X.chunk(i,col)[j] = 77;
      }

  PITTS::internal::HouseholderQR::transformBlock(nChunks, m, &X.chunk(0,0), X.colStrideChunks(), &Xresult.chunk(0,0), Xresult.colStrideChunks(), 2+nChunks, false);

  // check that the result is upper triangular, copied to the bottom
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
      if( i > nChunks*Chunk::size + j )
      {
        EXPECT_NEAR(RealType(0), std::real(Xresult(2*Chunk::size+i,j)), eps);
        EXPECT_NEAR(RealType(0), std::imag(Xresult(2*Chunk::size+i,j)), eps);
      }
      // X shouldn't change
      if( i < nChunks*Chunk::size )
      {
        ASSERT_EQ(X_ref(i,j), X(i,j));
      }
      else
      {
        ASSERT_EQ(Type(77), X(i,j));
      }
    }
  }

  // use Eigen to check that the singular values and the right singular vectors are identical
  auto mapXresult = ConstEigenMap(Xresult);
  //std::cout << "X:\n" << mapXresult << std::endl;
  auto mapX_ref = ConstEigenMap(X_ref);
#if EIGEN_VERSION_AT_LEAST(3,4,90)
  Eigen::BDCSVD<Eigen::MatrixX<Type>, Eigen::ComputeThinV> svd(mapXresult.bottomRows(n-nChunks*Chunk::size));
  Eigen::BDCSVD<Eigen::MatrixX<Type>, Eigen::ComputeThinV> svd_ref(mapX_ref);
#else
  Eigen::BDCSVD<Eigen::MatrixX<Type>> svd(mapXresult.bottomRows(n-nChunks*Chunk::size), Eigen::ComputeThinV);
  Eigen::BDCSVD<Eigen::MatrixX<Type>> svd_ref(mapX_ref, Eigen::ComputeThinV);
#endif

  ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  // V can differ by sign, only consider absolute part
  ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);
}


TYPED_TEST(PITTS_MultiVector_tsqr, internal_combineTwoBlocks)
{
  using Type = TestFixture::Type;
  using RealType = decltype(std::real(Type()));
  const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
  using Chunk = PITTS::Chunk<Type>;
  using MultiVector = PITTS::MultiVector<Type>;

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

    // create helper type
    MPI_Datatype mpiType;
    ASSERT_EQ(MPI_SUCCESS, MPI_Type_contiguous(totalSize, PITTS::internal::parallel::mpiType<Type>(), &mpiType));
    ASSERT_EQ(MPI_SUCCESS, MPI_Type_commit(&mpiType));
    int len = 1;

    PITTS::internal::HouseholderQR::combineTwoBlocks((const Type*)(&(buff1[0][0])), &(buff2[0][0]), &len, &mpiType);

    // remove helper type
    ASSERT_EQ(MPI_SUCCESS, MPI_Type_free(&mpiType));

    // compara singular values with Eigen
    Eigen::MatrixX<Type> R12(2*m,m);
    R12.block(0,0,m,m) = ConstEigenMap(R1);
    R12.block(m,0,m,m) = ConstEigenMap(R2);
    Eigen::JacobiSVD<Eigen::MatrixX<Type>> svd_ref(R12);

    Eigen::Map<Eigen::MatrixX<Type>> result(&(buff2[0][0]), mChunks*Chunk::size, m);
    Eigen::JacobiSVD<Eigen::MatrixX<Type>> svd(result);

    EXPECT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  }
}


namespace
{
  // helper function for testing block_TSQR with different data dimensions, etc
  template<typename T>
  void test_block_TSQR(int n, int m)
  {
    using RealType = decltype(std::real(T()));
    const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
    using Chunk = PITTS::Chunk<T>;
    using MultiVector = PITTS::MultiVector<T>;
    using Tensor2 = PITTS::Tensor2<T>;
    using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    MultiVector M(n,m);
    randomize(M);

    // store the original matrix for later
    EigenMatrix M_ref = ConstEigenMap(M);

    Tensor2 R;
    block_TSQR(M, R, 4, false);
    ASSERT_EQ(m, R.r1());
    ASSERT_EQ(m, R.r2());
    for(int j = 0; j < m; j++)
    {
      for(int i = j+1; i < m; i++)
      {
        ASSERT_NEAR(RealType(0), std::real(R(i,j)), eps);
        ASSERT_NEAR(RealType(0), std::imag(R(i,j)), eps);
      }
    }

    // check that the singular values and right singular vectors match...
#if EIGEN_VERSION_AT_LEAST(3,4,90)
    Eigen::BDCSVD<EigenMatrix, Eigen::ComputeThinV> svd(PITTS::ConstEigenMap(R));
    Eigen::BDCSVD<EigenMatrix, Eigen::ComputeThinV> svd_ref(M_ref);
#else
    Eigen::BDCSVD<EigenMatrix> svd(PITTS::ConstEigenMap(R), Eigen::ComputeThinV);
    Eigen::BDCSVD<EigenMatrix> svd_ref(M_ref, Eigen::ComputeThinV);
#endif

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

TYPED_TEST(PITTS_MultiVector_tsqr, block_TSQR_small_serial)
{
  using Type = TestFixture::Type;
  int nThreads = get_default_num_threads();

  omp_set_num_threads(1);
  test_block_TSQR<Type>(50, 10);
  omp_set_num_threads(nThreads);
}


TYPED_TEST(PITTS_MultiVector_tsqr, block_TSQR_small_4threads)
{
  using Type = TestFixture::Type;
  int nThreads = get_default_num_threads();

  ASSERT_LE(4, omp_get_max_threads());
  omp_set_num_threads(4);

  test_block_TSQR<Type>(50, 10);

  omp_set_num_threads(nThreads);
}


TYPED_TEST(PITTS_MultiVector_tsqr, block_TSQR_large_serial)
{
  using Type = TestFixture::Type;
  int nThreads = get_default_num_threads();

  omp_set_num_threads(1);
  test_block_TSQR<Type>(200, 30);
  omp_set_num_threads(nThreads);
}


TYPED_TEST(PITTS_MultiVector_tsqr, block_TSQR_large_varying_sizes_serial)
{
  using Type = TestFixture::Type;
  int nThreads = get_default_num_threads();

  omp_set_num_threads(1);
  for(int m = 1; m < 30; m++)
  {
    std::cout << "Testing #cols = " << m << "\n";
    test_block_TSQR<Type>(250, m);
  }
  omp_set_num_threads(nThreads);
}



TYPED_TEST(PITTS_MultiVector_tsqr, block_TSQR_large_parallel)
{
  using Type = TestFixture::Type;
  test_block_TSQR<Type>(200, 30);
}


TYPED_TEST(PITTS_MultiVector_tsqr, block_TSQR_manyRows_differentNumbersOfThreads)
{
  using Type = TestFixture::Type;
  int nThreads = get_default_num_threads();

  ASSERT_LE(4, omp_get_max_threads());
  for(int iThreads = 1; iThreads < 5; iThreads++)
  {
    omp_set_num_threads(iThreads);
    test_block_TSQR<Type>(1000, 1);
  }
  omp_set_num_threads(nThreads);
}


TYPED_TEST(PITTS_MultiVector_tsqr, block_TSQR_mpiGlobal)
{
  using Type = TestFixture::Type;
  using RealType = decltype(std::real(Type()));
  const RealType eps = std::sqrt(std::numeric_limits<RealType>::epsilon());
  using Chunk = PITTS::Chunk<Type>;
  using MultiVector = PITTS::MultiVector<Type>;
  using Tensor2 = PITTS::Tensor2<Type>;

  const long long nTotal = 500;
  const long long m = 4;

  const auto& [iProc,nProcs] = PITTS::internal::parallel::mpiProcInfo();
  const auto& [nFirst,nLast] = PITTS::internal::parallel::distribute(nTotal, {iProc,nProcs});
  const long long n = nLast - nFirst + 1;

  Eigen::MatrixX<Type> Mglobal = Eigen::MatrixX<Type>::Random(nTotal, 4);
  MPI_Bcast(Mglobal.data(), nTotal*m, PITTS::internal::parallel::mpiType<Type>(), 0, MPI_COMM_WORLD);

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
      ASSERT_NEAR(RealType(0), std::real(R(i,j)), eps);
      ASSERT_NEAR(RealType(0), std::imag(R(i,j)), eps);
    }
  }

  // check that the singular values and right singular vectors match...
#if EIGEN_VERSION_AT_LEAST(3,4,90)
  Eigen::BDCSVD<Eigen::MatrixX<Type>, Eigen::ComputeThinV> svd(ConstEigenMap(R));
  Eigen::BDCSVD<Eigen::MatrixX<Type>, Eigen::ComputeThinV> svd_ref(Mglobal);
#else
  Eigen::BDCSVD<Eigen::MatrixX<Type>> svd(ConstEigenMap(R), Eigen::ComputeThinV);
  Eigen::BDCSVD<Eigen::MatrixX<Type>> svd_ref(Mglobal, Eigen::ComputeThinV);
#endif

  ASSERT_NEAR(svd_ref.singularValues(), svd.singularValues(), eps);
  // V can differ by sign, only consider absolute part
  ASSERT_NEAR(svd_ref.matrixV().array().abs(), svd.matrixV().array().abs(), eps);
}
