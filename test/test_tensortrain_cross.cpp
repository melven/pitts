// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_cross.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_multivector.hpp"
#include "pitts_random.hpp"
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

  // evaluate a tensor train at one multi-index: the scalar product with the corresponding unit tensor
  template<typename T>
  T evaluate(const PITTS::TensorTrain<T>& TT, const Eigen::RowVectorXi& multiIndex)
  {
    PITTS::TensorTrain<T> unit(TT.dimensions());
    unit.setUnit(std::vector<int>(multiIndex.begin(), multiIndex.end()));
    return dot(unit, TT);
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

TYPED_TEST(PITTS_TensorTrain_cross, array_function)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  const std::vector<int> dims = {4, 3, 5, 3};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {2, 3, 2}, 5);

  // a function that is applied to all entries of a block at once must give the same result as the
  // corresponding scalar function (the algorithm only sees the same numbers); the random error index
  // sets require the same seed for both runs
  for(int nMALS: {1, 2})
  {
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;

    TensorTrain TTy(dims), TTyRef(dims);
    PITTS::internal::randomGenerator.seed(2026);
    const auto info = applyElementwise([](PITTS::MultiVector<Type>& mv)
    {
      for(long long i = 0; i < mv.rows(); i++)
        mv(i, 0) = std::sqrt(mv(i, 0));
    }, TTx, TTy, options);
    PITTS::internal::randomGenerator.seed(2026);
    const auto infoRef = applyElementwise([](Type v){return std::sqrt(v);}, TTx, TTyRef, options);

    EXPECT_EQ(infoRef.nEvaluations, info.nEvaluations);
    EXPECT_EQ(infoRef.nSweeps, info.nSweeps);
    ASSERT_EQ(TTyRef.getTTranks(), TTy.getTTranks());
    // only up to rounding differences of the threaded reductions inside the kernels
    const auto yRef = dense(TTyRef);
    EXPECT_NEAR(yRef, dense(TTy), tolerance<Type>()*yRef.norm());
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, convergence_measures)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  // mode dimensions and ranks vary strongly over the dimensions, so the blocks differ a lot in size:
  // the measures must still be comparable (rms scaling, one global reference)
  const std::vector<int> dims = {20, 2, 2, 2, 20};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {3, 2, 2, 3}, 23);

  TensorTrain TTy(dims);
  PITTS::TTCrossOptions<Type> options;
  options.rankTolerance = Type(1.e-8);
  options.nSweeps = 20;
  const auto info = applyElementwise([](Type v){return v*v;}, TTx, TTy, options);

  EXPECT_TRUE(info.converged);
  EXPECT_LT(info.lastChange, options.convergenceTolerance);
  EXPECT_LT(info.lastError, options.convergenceTolerance);
  // converged means in particular that the ranks did not grow in the last sweep
  const auto ranks = TTy.getTTranks();
  const auto infoAgain = applyElementwise([](Type v){return v*v;}, TTx, TTy, options);
  EXPECT_EQ(ranks, TTy.getTTranks());
  EXPECT_TRUE(infoAgain.converged);

  // scale invariance: scaling the function scales all blocks, the measures must not change
  TensorTrain TTscaled(dims);
  const auto infoScaled = applyElementwise([](Type v){return Type(1000)*v*v;}, TTx, TTscaled, options);
  EXPECT_NEAR(info.lastChange, infoScaled.lastChange, 10*tolerance<Type>());
  EXPECT_NEAR(info.lastError, infoScaled.lastError, 10*tolerance<Type>());

  // without enrichment there is no error estimate
  options.errorRank = 0;
  options.enrichmentRank = 0;
  TensorTrain TTnoEnrich(dims);
  const auto infoNoEnrich = applyElementwise([](Type v){return v*v;}, TTx, TTnoEnrich, options);
  EXPECT_EQ(Type(0), infoNoEnrich.lastError);

  // a run that is stopped while the ranks still grow is not converged
  options.errorRank = 3;
  options.enrichmentRank = 2;
  options.nSweeps = 1;
  TensorTrain TTshort(dims);
  const auto infoShort = applyElementwise([](Type v){return v*v;}, TTx, TTshort, options);
  EXPECT_FALSE(infoShort.converged);
}

TYPED_TEST(PITTS_TensorTrain_cross, control_sample)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  const std::vector<int> dims = {4, 3, 5, 3};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {2, 3, 2}, 31);
  const auto f = [](Type v){return v*v;};

  PITTS::TTCrossOptions<Type> options;
  options.rankTolerance = Type(1.e-8);
  options.nSweeps = 20;

  // same seed for both runs: the random error index sets (and thus the sweeps) are then identical,
  // so the difference in the number of evaluations is exactly the control sample
  PITTS::internal::randomGenerator.seed(31);
  TensorTrain TTy(dims);
  const auto info = applyElementwise(f, TTx, TTy, options);
  EXPECT_EQ(Type(0), info.lastSampleError);   // off by default

  // x^2 is recovered exactly, so the sampled error is at round-off level
  options.controlSampleSize = 50;
  PITTS::internal::randomGenerator.seed(31);
  TensorTrain TTsampled(dims);
  const auto infoSampled = applyElementwise(f, TTx, TTsampled, options);
  EXPECT_TRUE(infoSampled.converged);
  EXPECT_LT(infoSampled.lastSampleError, tolerance<Type>());
  // the control sample costs one function evaluation per sample and sweep
  EXPECT_EQ(info.nEvaluations + infoSampled.nSweeps*options.controlSampleSize, infoSampled.nEvaluations);

  // a deliberately bad approximation (rank 1, one sweep) must show up in the sampled error
  options.nSweeps = 1;
  options.maxRank = 1;
  options.errorRank = 0;
  options.enrichmentRank = 0;
  TensorTrain TTbad(dims);
  const auto infoBad = applyElementwise(f, TTx, TTbad, options);
  EXPECT_GT(infoBad.lastSampleError, Type(0.01));
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
