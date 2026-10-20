// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_symmetric.hpp"
#include "pitts_random.hpp"
#include "pitts_tensor3_random.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
  using PITTS::internal::symmetric_tt::leftOrthonormalizeHalf;
  using PITTS::internal::symmetric_tt::halfInnerProducts;
  using PITTS::internal::symmetric_tt::symmetricEigen;
  using PITTS::internal::symmetric_tt::symmetricRound;
  using PITTS::internal::symmetric_tt::positiveHalf;

  template<typename T>
  using mat = Eigen::MatrixX<T>;

  // random half: m cores, ranks[k] is the right rank of core k (the last one is the open bond)
  template<typename T>
  PITTS::TensorTrain<T> randomHalf(const std::vector<int>& dims, const std::vector<int>& ranks)
  {
    const int m = dims.size();
    std::vector<PITTS::Tensor3<T>> cores(m);
    for(int k = 0; k < m; k++)
    {
      cores[k].resize(k == 0 ? 1 : ranks[k-1], dims[k], ranks[k]);
      randomize(cores[k]);
    }
    return PITTS::TensorTrain<T>(std::move(cores));
  }

  // values of a half as an N x r matrix (row = i_0 + n_0 i_1 + ..., N = n_0 ... n_{m-1})
  template<typename T>
  mat<T> denseHalf(const PITTS::TensorTrain<T>& H)
  {
    mat<T> A = mat<T>::Ones(1, 1);
    for(int k = 0; k < int(H.dimensions().size()); k++)
    {
      const auto& core = H.subTensor(k);
      mat<T> next(A.rows() * core.n(), core.r2());
      for(int j = 0; j < core.n(); j++)
      {
        mat<T> slice(core.r1(), core.r2());
        for(int a = 0; a < core.r1(); a++)
          for(int b = 0; b < core.r2(); b++)
            slice(a, b) = core(a, j, b);
        // row of the left multi-index (i_0..i_{k-1}) plus j * (number of left multi-indices)
        next.middleRows(j * A.rows(), A.rows()) = A * slice;
      }
      A = std::move(next);
    }
    return A;
  }

  // y = A S A^T as a dense (symmetric) matrix
  template<typename T>
  mat<T> denseSymmetric(const PITTS::TensorTrain<T>& G, const PITTS::Tensor2<T>& S)
  {
    const mat<T> A = denseHalf(G);
    return A * ConstEigenMap(S) * A.transpose();
  }

  // random symmetric matrix with the given eigenvalues
  template<typename T>
  PITTS::Tensor2<T> symmetricWithEigenvalues(const std::vector<T>& lambda)
  {
    const int n = lambda.size();
    const mat<T> Q = Eigen::HouseholderQR<mat<T>>(mat<T>::Random(n, n)).householderQ();
    PITTS::Tensor2<T> S(n, n);
    EigenMap(S) = Q * Eigen::Map<const Eigen::VectorX<T>>(lambda.data(), n).asDiagonal() * Q.transpose();
    return S;
  }
}

template<typename T>
class PITTS_TensorTrain_symmetric : public ::testing::Test
{
  public:
    using Type = T;
};

using SymmetricTestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PITTS_TensorTrain_symmetric, SymmetricTestTypes);

TYPED_TEST(PITTS_TensorTrain_symmetric, leftOrthonormalizeHalf)
{
  using Type = typename TestFixture::Type;
  const auto eps = 100*std::numeric_limits<Type>::epsilon();

  PITTS::internal::randomGenerator.seed(31);   // same data on all MPI processes
  for(const std::vector<int>& dims: {std::vector<int>{3, 4, 2}, std::vector<int>{5}})
  {
    const int m = dims.size();
    const std::vector<int> ranks = m == 1 ? std::vector<int>{3} : std::vector<int>{2, 3, 5};
    const PITTS::TensorTrain<Type> H = randomHalf<Type>(dims, ranks);
    PITTS::TensorTrain<Type> G(dims);
    copy(H, G);
    const PITTS::Tensor2<Type> R = leftOrthonormalizeHalf(G);

    const mat<Type> A = denseHalf(G);
    ASSERT_EQ(R.r1(), A.cols());
    EXPECT_NEAR(mat<Type>::Identity(A.cols(), A.cols()), A.transpose() * A, eps);
    EXPECT_NEAR(denseHalf(H), A * ConstEigenMap(R), eps * denseHalf(H).norm());
    for(int k = 0; k < m; k++)
      EXPECT_EQ(PITTS::TT_Orthogonality::left, G.isOrthonormal(k));
  }

  // open bond larger than the number of rows: only N directions remain
  const PITTS::TensorTrain<Type> Hwide = randomHalf<Type>({2, 3}, {2, 10});
  PITTS::TensorTrain<Type> Gwide(std::vector<int>{2, 3});
  copy(Hwide, Gwide);
  const PITTS::Tensor2<Type> Rwide = leftOrthonormalizeHalf(Gwide);
  EXPECT_EQ(6, Rwide.r1());
  EXPECT_EQ(10, Rwide.r2());
  EXPECT_NEAR(denseHalf(Hwide), denseHalf(Gwide) * ConstEigenMap(Rwide), eps * denseHalf(Hwide).norm());
}

TYPED_TEST(PITTS_TensorTrain_symmetric, halfInnerProducts)
{
  using Type = typename TestFixture::Type;
  const auto eps = 100*std::numeric_limits<Type>::epsilon();

  PITTS::internal::randomGenerator.seed(37);
  const PITTS::TensorTrain<Type> A = randomHalf<Type>({3, 2, 4}, {2, 3, 4});
  const PITTS::TensorTrain<Type> B = randomHalf<Type>({3, 2, 4}, {3, 1, 2});
  const PITTS::Tensor2<Type> AtB = halfInnerProducts(A, B);
  const mat<Type> ref = denseHalf(A).transpose() * denseHalf(B);
  EXPECT_NEAR(ref, ConstEigenMap(AtB), eps * ref.norm());

  const PITTS::TensorTrain<Type> other = randomHalf<Type>({3, 2, 3}, {2, 3, 4});
  EXPECT_THROW(halfInnerProducts(A, other), std::invalid_argument);
}

TYPED_TEST(PITTS_TensorTrain_symmetric, symmetricEigen)
{
  using Type = typename TestFixture::Type;
  const auto eps = 100*std::numeric_limits<Type>::epsilon();

  std::srand(41);
  const std::vector<Type> lambda = {Type(-0.001), Type(2), Type(0.1), Type(-1), Type(0.01), Type(0.5)};
  const PITTS::Tensor2<Type> C = symmetricWithEigenvalues(lambda);

  // full decomposition: sorted by |lambda|, orthonormal, reconstructs C
  const auto full = symmetricEigen(C);
  const std::vector<Type> sorted = {Type(2), Type(-1), Type(0.5), Type(0.1), Type(0.01), Type(-0.001)};
  ASSERT_EQ(6, full.lambda.size());
  for(int i = 0; i < 6; i++)
    EXPECT_NEAR(sorted[i], full.lambda[i], eps);
  const mat<Type> U = ConstEigenMap(full.U);
  EXPECT_NEAR(mat<Type>::Identity(6, 6), U.transpose() * U, eps);
  const Eigen::VectorX<Type> l = Eigen::Map<const Eigen::VectorX<Type>>(full.lambda.data(), 6);
  EXPECT_NEAR(ConstEigenMap(C), U * l.asDiagonal() * U.transpose(), eps);

  // truncation in the Frobenius norm: the discarded part is within the tolerance, one more would not be
  const Type normC = ConstEigenMap(C).norm();
  const auto truncated = symmetricEigen(C, Type(0.06) / normC);
  EXPECT_EQ(4, truncated.lambda.size());   // drops 0.01 and -0.001 (norm 0.01005), not 0.1
  const auto absolute = symmetricEigen(C, Type(0.06), std::numeric_limits<int>::max(), true);
  EXPECT_EQ(4, absolute.lambda.size());
  EXPECT_EQ(2, symmetricEigen(C, Type(0), 2).lambda.size());
  EXPECT_EQ(1, symmetricEigen(C, Type(10)).lambda.size());   // at least one

  // round-off asymmetry is removed: the result is the one of the symmetric part
  PITTS::Tensor2<Type> Cnoise(6, 6);
  EigenMap(Cnoise) = ConstEigenMap(C);
  Cnoise(0, 1) += Type(1.e-3);
  Cnoise(1, 0) -= Type(1.e-3);
  const auto fromNoise = symmetricEigen(Cnoise);
  for(int i = 0; i < 6; i++)
    EXPECT_NEAR(full.lambda[i], fromNoise.lambda[i], eps);

  EXPECT_THROW(symmetricEigen(PITTS::Tensor2<Type>(2, 3)), std::invalid_argument);
}

TYPED_TEST(PITTS_TensorTrain_symmetric, symmetricRound)
{
  using Type = typename TestFixture::Type;
  const auto eps = 1000*std::numeric_limits<Type>::epsilon();

  PITTS::internal::randomGenerator.seed(43);
  std::srand(43);
  const std::vector<int> dims = {3, 4, 2};

  // left-orthonormal G and an indefinite S with decaying eigenvalues
  PITTS::TensorTrain<Type> G0 = randomHalf<Type>(dims, {3, 6, 6});
  leftOrthonormalizeHalf(G0);
  const PITTS::Tensor2<Type> S0 = symmetricWithEigenvalues<Type>({Type(1), Type(-0.5), Type(0.1), Type(0.01), Type(-0.001), Type(1.e-4)});
  const mat<Type> y = denseSymmetric(G0, S0);

  // orthogonality only up to eps * cond: normalize_svd/normalize_qb reconstruct the orthogonal factor from the Q-less TSQR
  // (Q = M V S^{-1}), and the singular values here span 4 orders of magnitude
  const auto orthoTol = Type(1.e5) * std::numeric_limits<Type>::epsilon();
  const auto checkGauge = [&](const PITTS::TensorTrain<Type>& G, const PITTS::Tensor2<Type>& S)
  {
    const mat<Type> A = denseHalf(G);
    EXPECT_NEAR(mat<Type>::Identity(A.cols(), A.cols()), A.transpose() * A, orthoTol);
    const mat<Type> Smat = ConstEigenMap(S);
    EXPECT_NEAR(mat<Type>(Smat.diagonal().asDiagonal()), Smat, eps);
    for(int i = 1; i < S.r1(); i++)
      EXPECT_LE(std::abs(S(i, i)), std::abs(S(i-1, i-1)));
  };

  // tolerance 0: same tensor
  {
    PITTS::TensorTrain<Type> G(dims);
    copy(G0, G);
    PITTS::Tensor2<Type> S(6, 6);
    EigenMap(S) = ConstEigenMap(S0);
    symmetricRound(G, S, Type(0));
    checkGauge(G, S);
    EXPECT_NEAR(y, denseSymmetric(G, S), eps);
  }

  // truncation: error within the tolerance, smaller ranks
  for(const Type tol: {Type(1.e-3), Type(1.e-2), Type(0.2)})
  {
    PITTS::TensorTrain<Type> G(dims);
    copy(G0, G);
    PITTS::Tensor2<Type> S(6, 6);
    EigenMap(S) = ConstEigenMap(S0);
    symmetricRound(G, S, tol);
    checkGauge(G, S);
    EXPECT_LE((y - denseSymmetric(G, S)).norm(), tol * y.norm());
    EXPECT_LT(S.r1(), 6);
  }

  // maximal rank
  {
    PITTS::TensorTrain<Type> G(dims);
    copy(G0, G);
    PITTS::Tensor2<Type> S(6, 6);
    EigenMap(S) = ConstEigenMap(S0);
    symmetricRound(G, S, Type(0), 2);
    checkGauge(G, S);
    EXPECT_EQ(2, S.r1());
    for(int r: G.getTTranks())
      EXPECT_LE(r, 2);
  }

  // a low-rank tensor in a larger frame: rounding finds the ranks of the low-rank half
  {
    const PITTS::TensorTrain<Type> Hsmall = randomHalf<Type>(dims, {2, 2, 2});
    PITTS::TensorTrain<Type> G(dims);
    copy(Hsmall, G);
    const PITTS::Tensor2<Type> R = leftOrthonormalizeHalf(G);
    // y = Hsmall Hsmall^T = G (R R^T) G^T, padded with zero directions on the open bond
    PITTS::TensorTrain<Type> Gpad = randomHalf<Type>(dims, {3, 5, 4});
    // put G into the first block of Gpad and orthonormalize: span contains span(G)
    std::vector<PITTS::Tensor3<Type>> cores(3);
    for(int k = 0; k < 3; k++)
    {
      const auto& small = G.subTensor(k);
      const auto& big = Gpad.subTensor(k);
      cores[k].resize(k == 0 ? 1 : small.r1() + big.r1(), dims[k], small.r2() + big.r2());
      cores[k].setConstant(Type(0));
      for(int a = 0; a < small.r1(); a++)
        for(int j = 0; j < dims[k]; j++)
          for(int b = 0; b < small.r2(); b++)
            cores[k](a, j, b) = small(a, j, b);
      for(int a = 0; a < big.r1(); a++)
        for(int j = 0; j < dims[k]; j++)
          for(int b = 0; b < big.r2(); b++)
            cores[k](k == 0 ? 0 : small.r1()+a, j, small.r2()+b) = big(a, j, b);
    }
    PITTS::TensorTrain<Type> Gbig(std::move(cores));
    const PITTS::Tensor2<Type> Rbig = leftOrthonormalizeHalf(Gbig);
    // S = Rbig P (R R^T) P^T Rbig^T with P the embedding of the first block
    mat<Type> P = mat<Type>::Zero(Rbig.r2(), 2);
    P.topRows(2) = mat<Type>::Identity(2, 2);
    PITTS::Tensor2<Type> S(Rbig.r1(), Rbig.r1());
    EigenMap(S) = ConstEigenMap(Rbig) * P * ConstEigenMap(R) * ConstEigenMap(R).transpose() * P.transpose() * ConstEigenMap(Rbig).transpose();
    const mat<Type> ySmall = denseSymmetric(Gbig, S);
    symmetricRound(Gbig, S, Type(std::sqrt(std::numeric_limits<Type>::epsilon())));
    checkGauge(Gbig, S);
    EXPECT_EQ(2, S.r1());
    for(int r: Gbig.getTTranks())
      EXPECT_LE(r, 2);
    EXPECT_NEAR(ySmall, denseSymmetric(Gbig, S), std::sqrt(std::numeric_limits<Type>::epsilon()) * ySmall.norm());
  }

  // a single core (d = 2): only the middle bond
  {
    PITTS::TensorTrain<Type> G = randomHalf<Type>({5}, {5});
    leftOrthonormalizeHalf(G);
    PITTS::Tensor2<Type> S = symmetricWithEigenvalues<Type>({Type(1), Type(-0.1), Type(1.e-3), Type(1.e-4), Type(0)});
    const mat<Type> y1 = denseSymmetric(G, S);
    symmetricRound(G, S, Type(0.01));
    checkGauge(G, S);
    EXPECT_EQ(2, S.r1());
    EXPECT_LE((y1 - denseSymmetric(G, S)).norm(), Type(0.01) * y1.norm());
  }

  PITTS::Tensor2<Type> wrong(3, 3);
  PITTS::TensorTrain<Type> G(dims);
  copy(G0, G);
  EXPECT_THROW(symmetricRound(G, wrong, Type(0)), std::invalid_argument);
}

TYPED_TEST(PITTS_TensorTrain_symmetric, positiveHalf)
{
  using Type = typename TestFixture::Type;
  const auto eps = 1000*std::numeric_limits<Type>::epsilon();

  PITTS::internal::randomGenerator.seed(47);
  std::srand(47);
  const std::vector<int> dims = {2, 3};
  PITTS::TensorTrain<Type> G = randomHalf<Type>(dims, {2, 4});
  leftOrthonormalizeHalf(G);
  PITTS::TensorTrain<Type> H(dims);

  // positive semidefinite: H H^T = G S G^T
  const PITTS::Tensor2<Type> Spsd = symmetricWithEigenvalues<Type>({Type(2), Type(1), Type(0.5), Type(0.25)});
  EXPECT_NEAR(Type(0), positiveHalf(G, Spsd, H), eps);
  EXPECT_EQ(4, H.subTensor(1).r2());
  mat<Type> AH = denseHalf(H);
  EXPECT_NEAR(denseSymmetric(G, Spsd), AH * AH.transpose(), eps);

  // indefinite: the negative eigenvalues are removed, their norm is returned
  const PITTS::Tensor2<Type> Sind = symmetricWithEigenvalues<Type>({Type(2), Type(-0.3), Type(0.5), Type(-0.4)});
  EXPECT_NEAR(Type(0.5), positiveHalf(G, Sind, H), eps);
  EXPECT_EQ(2, H.subTensor(1).r2());
  AH = denseHalf(H);
  const auto eig = symmetricEigen(Sind);
  mat<Type> SindPlus = mat<Type>::Zero(4, 4);
  for(int j = 0; j < 4; j++)
    if( eig.lambda[j] > 0 )
      SindPlus += eig.lambda[j] * ConstEigenMap(eig.U).col(j) * ConstEigenMap(eig.U).col(j).transpose();
  const mat<Type> A = denseHalf(G);
  EXPECT_NEAR(mat<Type>(A * SindPlus * A.transpose()), mat<Type>(AH * AH.transpose()), eps);

  // negative definite: zero half of rank 1
  const PITTS::Tensor2<Type> Sneg = symmetricWithEigenvalues<Type>({Type(-3), Type(-4), Type(-0.1), Type(-0.1)});
  EXPECT_NEAR(ConstEigenMap(Sneg).norm(), positiveHalf(G, Sneg, H), eps);
  EXPECT_EQ(1, H.subTensor(1).r2());
  EXPECT_NEAR(Type(0), denseHalf(H).norm(), eps);
}

TYPED_TEST(PITTS_TensorTrain_symmetric, rightOrthonormalHalf)
{
  using Type = typename TestFixture::Type;
  const auto eps = 1000*std::numeric_limits<Type>::epsilon();

  PITTS::internal::randomGenerator.seed(53);
  std::srand(53);
  const std::vector<int> dims = {3, 2, 4};
  PITTS::TensorTrain<Type> G = randomHalf<Type>(dims, {3, 4, 3});
  leftOrthonormalizeHalf(G);
  const PITTS::Tensor2<Type> S = symmetricWithEigenvalues<Type>({Type(2), Type(-1), Type(0.5)});
  PITTS::TensorTrain<Type> Y(dims);
  PITTS::internal::symmetric_tt::rightOrthonormalHalf(G, S, Y);

  // the same half G S, with right-orthonormal cores 1..m-1
  const mat<Type> ref = denseHalf(G) * ConstEigenMap(S);
  EXPECT_NEAR(ref, denseHalf(Y), eps * ref.norm());
  for(int k = 1; k < 3; k++)
  {
    const auto& core = Y.subTensor(k);
    const mat<Type> Vt = ConstEigenMap(unfold_right(core));
    EXPECT_NEAR(mat<Type>::Identity(core.r1(), core.r1()), mat<Type>(Vt * Vt.transpose()), eps);
  }
  EXPECT_THROW(PITTS::internal::symmetric_tt::rightOrthonormalHalf(G, PITTS::Tensor2<Type>(2, 2), Y), std::invalid_argument);
}
