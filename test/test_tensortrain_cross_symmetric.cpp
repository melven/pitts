// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_cross.hpp"
#include "pitts_random.hpp"
#include "pitts_tensor3_random.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
  template<typename T>
  using mat = Eigen::MatrixX<T>;

  // random half: m cores, ranks[k] is the right rank of core k (the last one is the open bond)
  template<typename T>
  PITTS::TensorTrain<T> randomHalf(const std::vector<int>& dims, const std::vector<int>& ranks, T scale = T(1))
  {
    const int m = dims.size();
    std::vector<PITTS::Tensor3<T>> cores(m);
    for(int k = 0; k < m; k++)
    {
      cores[k].resize(k == 0 ? 1 : ranks[k-1], dims[k], ranks[k]);
      randomize(cores[k]);
      PITTS::internal::t3_scale(scale, cores[k]);
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
        next.middleRows(j * A.rows(), A.rows()) = A * slice;
      }
      A = std::move(next);
    }
    return A;
  }

  // the symmetric tensor H H^T as a matrix (rows (i_0..i_{m-1}), columns the mirrored (i_{2m-1}..i_m)); an element-wise
  // function of it is the element-wise function of this matrix
  template<typename T>
  mat<T> denseSymmetric(const PITTS::TensorTrain<T>& H)
  {
    const mat<T> A = denseHalf(H);
    return A * A.transpose();
  }

  template<typename T>
  T relativeError(const mat<T>& ref, const mat<T>& approx)
  {
    return (ref - approx).norm() / ref.norm();
  }
}

template<typename T>
class PITTS_TensorTrain_cross_symmetric : public ::testing::Test
{
  public:
    using Type = T;

    // tolerance for results that are exact up to round-off and the truncation tolerance of the algorithm
    static T exactTolerance() {return std::sqrt(std::numeric_limits<T>::epsilon()) * T(100);}
};

using SymmetricCrossTestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PITTS_TensorTrain_cross_symmetric, SymmetricCrossTestTypes);

TYPED_TEST(PITTS_TensorTrain_cross_symmetric, identity_and_square_exact)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type tol = TestFixture::exactTolerance();

  PITTS::internal::randomGenerator.seed(101);   // same data on all MPI processes
  for(const std::vector<int>& dims: {std::vector<int>{4}, std::vector<int>{3, 2}, std::vector<int>{2, 3, 2}})
  {
    const int m = dims.size();
    const std::vector<int> ranks = m == 1 ? std::vector<int>{2} : m == 2 ? std::vector<int>{2, 3} : std::vector<int>{2, 3, 2};
    const TensorTrain H = randomHalf<Type>(dims, ranks);
    const mat<Type> x = denseSymmetric(H);
    for(int nMALS: {1, 2})
    {
      PITTS::TTCrossOptions<Type> options;
      options.nMALS = nMALS;
      TensorTrain TTy(dims);

      // identity: y = x (positive semidefinite, nothing removed)
      auto info = applyElementwiseSymmetric([](Type v){return v;}, H, TTy, options);
      EXPECT_LT(relativeError(x, denseSymmetric(TTy)), tol);
      EXPECT_LT(info.lastNegativePart, tol);
      EXPECT_TRUE(info.converged);
      EXPECT_EQ(m, info.leftIndices.size());

      // square: y = x .* x (the Hadamard square of a Gram matrix is positive semidefinite)
      info = applyElementwiseSymmetric([](Type v){return v*v;}, H, TTy, options);
      EXPECT_LT(relativeError(mat<Type>(x.cwiseProduct(x)), denseSymmetric(TTy)), tol);
      EXPECT_LT(info.lastNegativePart, tol);
      EXPECT_TRUE(info.converged);
    }
  }
}

TYPED_TEST(PITTS_TensorTrain_cross_symmetric, exp_kernel)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  // an exponential kernel of a Gram matrix (positive semidefinite by the Schur product theorem), not of low rank
  PITTS::internal::randomGenerator.seed(103);
  const std::vector<int> dims = {3, 4, 3};
  const TensorTrain H = randomHalf<Type>(dims, {2, 3, 3}, Type(0.4));
  const mat<Type> x = denseSymmetric(H);
  const mat<Type> ref = x.array().exp().matrix();
  const auto f = [](Type v){return std::exp(v);};

  for(int nMALS: {1, 2})
    for(Type oversampling: {Type(0), Type(0.5)})
    {
      PITTS::TTCrossOptions<Type> options;
      options.nMALS = nMALS;
      options.oversampling = oversampling;
      options.nSweeps = 20;
      options.controlSampleSize = 200;
      TensorTrain TTy(dims);
      const auto info = applyElementwiseSymmetric(f, H, TTy, options);
      const Type error = relativeError(ref, denseSymmetric(TTy));
      EXPECT_LT(error, 100 * std::real(options.rankTolerance));
      EXPECT_LT(info.lastSampleError, 100 * std::real(options.rankTolerance));
      EXPECT_LT(info.lastNegativePart, 100 * std::real(options.rankTolerance));
      // (no check of info.converged: for a function without an exact low rank, the convergence measures may fluctuate
      // around the tolerance when a singular value is close to the truncation threshold, as for applyElementwise)
    }

  // other options: random error crosses, 2-site without error crosses
  PITTS::TTCrossOptions<Type> options;
  options.nSweeps = 20;
  options.randomErrorIndexSets = true;
  TensorTrain TTy(dims);
  applyElementwiseSymmetric(f, H, TTy, options);
  EXPECT_LT(relativeError(ref, denseSymmetric(TTy)), 100 * std::real(options.rankTolerance));
  // no enrichment: only the 2-site variant adapts the inner ranks
  options.randomErrorIndexSets = false;
  options.errorRank = 0;
  options.enrichmentRank = 0;
  options.nMALS = 2;
  applyElementwiseSymmetric(f, H, TTy, options);
  EXPECT_LT(relativeError(ref, denseSymmetric(TTy)), 100 * std::real(options.rankTolerance));
}

TYPED_TEST(PITTS_TensorTrain_cross_symmetric, indefinite_with_middle_matrix)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type tol = TestFixture::exactTolerance();

  // x - 1 (x minus the all-ones tensor) is indefinite with exact rank r + 1 at every bond
  PITTS::internal::randomGenerator.seed(107);
  const std::vector<int> dims = {2, 3, 2};
  const TensorTrain H = randomHalf<Type>(dims, {2, 3, 2});
  const mat<Type> x = denseSymmetric(H);
  const mat<Type> ref = (x.array() - Type(1)).matrix();
  const auto f = [](Type v){return v - Type(1);};

  for(int nMALS: {1, 2})
  {
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;
    TensorTrain G(dims);
    PITTS::Tensor2<Type> S;
    const auto info = applyElementwiseSymmetric(f, H, G, options, &S);
    const mat<Type> A = denseHalf(G);
    EXPECT_LT(relativeError(ref, mat<Type>(A * ConstEigenMap(S) * A.transpose())), tol);
    EXPECT_NEAR(mat<Type>::Identity(A.cols(), A.cols()), mat<Type>(A.transpose() * A), tol);
    EXPECT_EQ(Type(0), info.lastNegativePart);

    // positive semidefinite result: the negative part is removed and reported
    TensorTrain TTy(dims);
    const auto infoPsd = applyElementwiseSymmetric(f, H, TTy, options);
    const Eigen::SelfAdjointEigenSolver<mat<Type>> eig(ref);
    const Type negative = eig.eigenvalues().cwiseMin(Type(0)).norm() / ref.norm();
    EXPECT_NEAR(negative, infoPsd.lastNegativePart, tol);
    const mat<Type> refPlus = eig.eigenvectors() * eig.eigenvalues().cwiseMax(Type(0)).asDiagonal() * eig.eigenvectors().transpose();
    EXPECT_LT(relativeError(refPlus, denseSymmetric(TTy)), tol);
  }
}

TYPED_TEST(PITTS_TensorTrain_cross_symmetric, two_inputs_product)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type tol = TestFixture::exactTolerance();

  // the element-wise product of two Gram matrices: positive semidefinite with the product of the ranks
  PITTS::internal::randomGenerator.seed(109);
  const std::vector<int> dims = {3, 2, 3};
  const TensorTrain Ha = randomHalf<Type>(dims, {2, 3, 2}), Hb = randomHalf<Type>(dims, {3, 2, 2});
  const mat<Type> ref = denseSymmetric(Ha).cwiseProduct(denseSymmetric(Hb));
  for(int nMALS: {1, 2})
  {
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;
    options.nSweeps = 20;
    TensorTrain TTy(dims);
    const auto info = applyElementwiseSymmetric([](Type a, Type b){return a*b;}, Ha, Hb, TTy, options);
    EXPECT_LT(relativeError(ref, denseSymmetric(TTy)), tol);
    EXPECT_TRUE(info.converged);
  }
}

TYPED_TEST(PITTS_TensorTrain_cross_symmetric, maxRank_and_argument_checks)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  PITTS::internal::randomGenerator.seed(113);
  const std::vector<int> dims = {3, 4, 3};
  const TensorTrain H = randomHalf<Type>(dims, {2, 3, 3}, Type(0.4));
  const auto f = [](Type v){return std::exp(v);};
  PITTS::TTCrossOptions<Type> options;
  options.maxRank = 3;
  TensorTrain TTy(dims);
  applyElementwiseSymmetric(f, H, TTy, options);
  for(int r: TTy.getTTranks())
    EXPECT_LE(r, 3);
  EXPECT_LE(TTy.subTensor(2).r2(), 3);

  // mismatching dimensions, first rank != 1
  TensorTrain wrongDims(std::vector<int>{3, 4, 2});
  EXPECT_THROW(applyElementwiseSymmetric(f, H, wrongDims), std::invalid_argument);
  std::vector<PITTS::Tensor3<Type>> cores(3);
  for(int k = 0; k < 3; k++)
  {
    cores[k].resize(2, dims[k], 2);
    randomize(cores[k]);
  }
  const TensorTrain wrongRank(std::move(cores));
  EXPECT_THROW(applyElementwiseSymmetric(f, wrongRank, TTy), std::invalid_argument);
}
