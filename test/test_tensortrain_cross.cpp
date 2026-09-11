// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_cross.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace
{
  // random TT with entries of all cores in (0.5, 1.5): all tensor entries are positive (for sqrt, log, 1/x)
  template<typename T>
  PITTS::TensorTrain<T> positiveRandomTT(const std::vector<int>& dims, const std::vector<int>& ranks, unsigned seed)
  {
    PITTS::TensorTrain<T> TT(dims);
    if( !ranks.empty() )
      TT.setTTranks(ranks);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dist(0.5, 1.5);
    for(int k = 0; k < dims.size(); k++)
    {
      TT.editSubTensor(k, [&](PITTS::Tensor3<T>& X)
      {
        for(int i = 0; i < X.r1(); i++)
          for(int j = 0; j < X.n(); j++)
            for(int l = 0; l < X.r2(); l++)
              X(i, j, l) = dist(gen);
      });
    }
    return TT;
  }

  // dense tensor as a vector (ordering of toDense, the same for x and y)
  template<typename T>
  Eigen::Matrix<T, Eigen::Dynamic, 1> dense(const PITTS::TensorTrain<T>& TT)
  {
    PITTS::MultiVector<T> mv;
    toDense(TT, mv);
    return ConstEigenMap(mv).col(0);
  }

  // evaluate a TT at one multi-index (core slice products), for the interpolation property
  template<typename T>
  T evaluate(const PITTS::TensorTrain<T>& TT, const Eigen::RowVectorXi& multiIndex)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    mat v = mat::Identity(1, 1);
    for(int k = 0; k < multiIndex.size(); k++)
    {
      const auto& X = TT.subTensor(k);
      mat slice(X.r1(), X.r2());
      for(int i = 0; i < X.r1(); i++)
        for(int j = 0; j < X.r2(); j++)
          slice(i, j) = X(i, multiIndex(k), j);
      v = v * slice;
    }
    return v(0, 0);
  }

  template<typename T>
  auto tolerance()
  {
    return 100*std::sqrt(std::numeric_limits<T>::epsilon());
  }
}

template<typename T>
class PITTS_TensorTrain_cross : public ::testing::Test
{
  public:
    using Type = T;
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PITTS_TensorTrain_cross, TestTypes);


TYPED_TEST(PITTS_TensorTrain_cross, invalid_arguments)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  const std::vector<int> dims = {3, 4, 2};
  const std::vector<int> wrongDims = {3, 4, 3};
  TensorTrain TTx(dims, 2), TTy(dims), TTwrong(wrongDims);
  randomize(TTx);
  const auto id = [](Type v){return v;};

  EXPECT_THROW(applyElementwise(id, TTx, TTwrong), std::invalid_argument);

  PITTS::TTCrossOptions<Type> options;
  options.nMALS = 3;
  EXPECT_THROW(applyElementwise(id, TTx, TTy, options), std::invalid_argument);
  options.nMALS = 1;
  options.enrichmentRank = 5;
  options.errorRank = 3;
  EXPECT_THROW(applyElementwise(id, TTx, TTy, options), std::invalid_argument);

  // boundary ranks != 1
  std::vector<PITTS::Tensor3<Type>> cores(2);
  cores[0].resize(2, 3, 2);
  cores[1].resize(2, 4, 1);
  const std::vector<int> dims2 = {3, 4};
  TensorTrain TTboundary(std::move(cores)), TTy2(dims2);
  randomize(TTboundary);
  EXPECT_THROW(applyElementwise(id, TTboundary, TTy2), std::invalid_argument);
}

TYPED_TEST(PITTS_TensorTrain_cross, identity_exact_recovery)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const auto eps = tolerance<Type>();

  const std::vector<int> dims = {3, 4, 2, 5};
  TensorTrain TTx(dims);
  TTx.setTTranks({2, 3, 2});
  randomize(TTx);
  const auto xDense = dense(TTx);

  for(int nMALS: {1, 2})
  {
    TensorTrain TTy(dims);
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;
    options.rankTolerance = Type(1.e-6);
    const auto info = applyElementwise([](Type v){return v;}, TTx, TTy, options);

    EXPECT_TRUE(info.converged);
    EXPECT_EQ(TTx.getTTranks(), TTy.getTTranks());
    EXPECT_GT(info.nEvaluations, 0);
    EXPECT_NEAR(xDense, dense(TTy), eps*xDense.norm());
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, square_and_sqrt_vs_dense)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const auto eps = tolerance<Type>();

  const std::vector<int> dims = {4, 3, 5, 3};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {2, 3, 2}, 42);
  const auto xDense = dense(TTx);

  for(int nMALS: {1, 2})
  {
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;
    options.rankTolerance = Type(1.e-7);
    options.nSweeps = 20;

    // x^2 has exact TT ranks <= r^2 (Hadamard product), so this is recovered exactly
    TensorTrain TTy(dims);
    const auto infoSquare = applyElementwise([](Type v){return v*v;}, TTx, TTy, options);
    EXPECT_TRUE(infoSquare.converged);
    const Eigen::Matrix<Type, Eigen::Dynamic, 1> squareDense = xDense.array().square();
    EXPECT_NEAR(squareDense, dense(TTy), eps*squareDense.norm());
    for(int k = 0; k < dims.size()-1; k++)
      EXPECT_LE(TTy.getTTranks()[k], TTx.getTTranks()[k]*TTx.getTTranks()[k]);

    // sqrt(x) is not exactly low-rank, but well approximable for entries in a moderate range
    TensorTrain TTz(dims);
    const auto infoSqrt = applyElementwise([](Type v){return std::sqrt(v);}, TTx, TTz, options);
    const Eigen::Matrix<Type, Eigen::Dynamic, 1> sqrtDense = xDense.array().sqrt();
    EXPECT_NEAR(sqrtDense, dense(TTz), 100*eps*sqrtDense.norm());
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, interpolation_property_on_final_crosses)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const auto eps = tolerance<Type>();

  const std::vector<int> dims = {3, 4, 3, 2};
  const int nDim = dims.size();
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {2, 2, 2}, 7);
  const auto f = [](Type v){return std::exp(v);};

  for(int nMALS: {1, 2})
  {
    TensorTrain TTy(dims);
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;
    options.rankTolerance = Type(0); // no truncation: the interpolation property holds exactly
    const auto info = applyElementwise(f, TTx, TTy, options);

    ASSERT_EQ(nDim-1, info.leftIndices.size());
    ASSERT_EQ(nDim, info.rightIndices.size());
    const auto ranks = TTy.getTTranks();
    for(int k = 0; k < nDim-1; k++)
    {
      // right sets of bond k are rightIndices[k+1] and match the final ranks;
      // the left sets may contain up to enrichmentRank additional indices (nMALS = 1)
      ASSERT_EQ(ranks[k], info.rightIndices[k+1].rows());
      ASSERT_EQ(nDim-k-1, info.rightIndices[k+1].cols());
      ASSERT_LE(ranks[k], info.leftIndices[k].rows());
      ASSERT_LE(info.leftIndices[k].rows(), ranks[k] + (nMALS == 1 ? options.enrichmentRank : 0));
      ASSERT_EQ(k+1, info.leftIndices[k].cols());

      // y interpolates f(x) on the fibers (I_{k-1}, S_k, J_{k+1}): check the crosses (I_k, J_{k+1})
      for(int a = 0; a < info.leftIndices[k].rows(); a++)
        for(int b = 0; b < info.rightIndices[k+1].rows(); b++)
        {
          Eigen::RowVectorXi multiIndex(nDim);
          multiIndex << info.leftIndices[k].row(a), info.rightIndices[k+1].row(b);
          const Type ref = f(evaluate(TTx, multiIndex));
          EXPECT_NEAR(ref, evaluate(TTy, multiIndex), eps*std::abs(ref));
        }
    }
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, qtt_sqrt_with_rank_growth)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const auto eps = tolerance<Type>();

  // QTT-like: small mode sizes, many dimensions; 1/x needs a larger rank than x
  const std::vector<int> dims(8, 2);
  const TensorTrain TTx = positiveRandomTT<Type>(dims, std::vector<int>(7, 2), 3);
  const auto xDense = dense(TTx);
  const Eigen::Matrix<Type, Eigen::Dynamic, 1> invDense = xDense.array().inverse();

  for(int nMALS: {1, 2})
  {
    TensorTrain TTy(dims);
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;
    options.rankTolerance = Type(1.e-6);
    options.nSweeps = 30;
    options.maxRank = 40;
    const auto info = applyElementwise([](Type v){return Type(1)/v;}, TTx, TTy, options);
    EXPECT_NEAR(invDense, dense(TTy), 100*eps*invDense.norm());
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, enrichment_options)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const auto eps = tolerance<Type>();

  const std::vector<int> dims = {3, 3, 3, 3, 3};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {2, 2, 2, 2}, 11);
  const auto xDense = dense(TTx);
  const Eigen::Matrix<Type, Eigen::Dynamic, 1> ref = xDense.array().square();

  // without enrichment the ALS-cross cannot increase the ranks (start ranks 2, x^2 needs up to 4)
  {
    TensorTrain TTy(dims);
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = 1;
    options.errorRank = 0;
    options.enrichmentRank = 0;
    applyElementwise([](Type v){return v*v;}, TTx, TTy, options);
    for(int r: TTy.getTTranks())
      EXPECT_LE(r, 2);
  }

  // with enrichment (random and derived error index sets) the exact result is found
  for(bool random: {true, false})
  {
    TensorTrain TTy(dims);
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = 1;
    options.errorRank = 2;
    options.enrichmentRank = 2;
    options.randomErrorIndexSets = random;
    options.rankTolerance = Type(1.e-7);
    options.nSweeps = 20;
    const auto info = applyElementwise([](Type v){return v*v;}, TTx, TTy, options);
    EXPECT_TRUE(info.converged);
    EXPECT_NEAR(ref, dense(TTy), eps*ref.norm());
  }
}
