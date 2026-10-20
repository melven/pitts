// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include "pitts_tensortrain_cross.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_from_dense_classical.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_multivector.hpp"
#include "pitts_random.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "eigen_test_helper.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
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
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

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
    options.rankTolerance = eps;
    const auto info = applyElementwise([](Type v){return v;}, TTx, TTy, options);

    EXPECT_TRUE(info.converged);
    EXPECT_EQ(TTx.getTTranks(), TTy.getTTranks());
    EXPECT_GT(info.nEvaluations, 0);
    EXPECT_NEAR(xDense, dense(TTy), 100*eps*xDense.norm());
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, square_and_sqrt_vs_dense)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  const std::vector<int> dims = {4, 3, 5, 3};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {2, 3, 2}, 42);
  const auto xDense = dense(TTx);

  for(int nMALS: {1, 2})
  {
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;
    options.rankTolerance = eps;
    options.nSweeps = 20;

    // x^2 has exact TT ranks <= r^2 (Hadamard product), so this is recovered exactly
    TensorTrain TTy(dims);
    const auto infoSquare = applyElementwise([](Type v){return v*v;}, TTx, TTy, options);
    EXPECT_TRUE(infoSquare.converged);
    const Eigen::Matrix<Type, Eigen::Dynamic, 1> squareDense = xDense.array().square();
    EXPECT_NEAR(squareDense, dense(TTy), 100*eps*squareDense.norm());
    for(int k = 0; k < dims.size()-1; k++)
      EXPECT_LE(TTy.getTTranks()[k], TTx.getTTranks()[k]*TTx.getTTranks()[k]);

    // sqrt(x) is not exactly low-rank, but well approximable for entries in a moderate range
    TensorTrain TTz(dims);
    const auto infoSqrt = applyElementwise([](Type v){return std::sqrt(v);}, TTx, TTz, options);
    const Eigen::Matrix<Type, Eigen::Dynamic, 1> sqrtDense = xDense.array().sqrt();
    EXPECT_NEAR(sqrtDense, dense(TTz), 10000*eps*sqrtDense.norm());
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, interpolation_property_on_final_crosses)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

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
          EXPECT_NEAR(ref, evaluate(TTy, multiIndex), 100*eps*std::abs(ref));
        }
    }
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, qtt_sqrt_with_rank_growth)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

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
    options.rankTolerance = eps;
    options.nSweeps = 30;
    options.maxRank = 40;
    const auto info = applyElementwise([](Type v){return Type(1)/v;}, TTx, TTy, options);
    EXPECT_NEAR(invDense, dense(TTy), 10000*eps*invDense.norm());
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, maxRank_limits_ranks_and_index_sets)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  const std::vector<int> dims = {3, 4, 4, 3};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {2, 3, 2}, 11);
  const auto f = [](Type v){return v*v*v;};

  PITTS::TTCrossOptions<Type> options;
  options.rankTolerance = Type(0);  // only maxRank truncates
  options.maxRank = 2;
  // the limit also applies to the 2-site SVD (nMALS = 2) and after the enrichment (nMALS = 1),
  // the oversampled index sets have at most maxRank + ceil(oversampling maxRank) multi-indices
  for(int nMALS: {1, 2})
    for(Type oversampling: {Type(0), Type(0.5)})
    {
      options.nMALS = nMALS;
      options.oversampling = oversampling;
      TensorTrain TTy(dims);
      const auto info = applyElementwise(f, TTx, TTy, options);
      // x^3 has the full ranks (3, 12, 3) (Hadamard product: at most r^3)
      EXPECT_EQ(std::vector<int>({2, 2, 2}), TTy.getTTranks());
      const int maxIndices = options.maxRank + static_cast<int>(std::ceil(oversampling * options.maxRank));
      for(int k = 0; k < 3; k++)
      {
        EXPECT_LE(info.leftIndices[k].rows(), maxIndices);
        EXPECT_LE(info.rightIndices[k+1].rows(), maxIndices);
      }
    }

  // a small bond also limits its neighbors, the ranks and the oversampled index sets (only by the dimensions here):
  // bond 1 of dims (6, 6, 2) allows rank n_2 = 2, so the oversampled limit 2 + ceil(0.5*2) = 3 is reduced to 2
  const std::vector<int> dims2 = {6, 6, 2};
  const TensorTrain TTx2 = positiveRandomTT<Type>(dims2, {3, 2}, 12);
  for(int nMALS: {1, 2})
  {
    PITTS::TTCrossOptions<Type> options2;
    options2.nMALS = nMALS;
    options2.oversampling = Type(0.5);
    options2.nSweeps = 3;
    TensorTrain TTy2(dims2);
    const auto info = applyElementwise([](Type v){return Type(1)/(Type(1)+v);}, TTx2, TTy2, options2);
    EXPECT_LE(TTy2.getTTranks()[1], 2);
    EXPECT_LE(info.leftIndices[1].rows(), 2);
    EXPECT_LE(info.leftIndices[0].rows(), 6);
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, rankTolerance_bounds_the_error)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  // y is kept in the orthogonal gauge, so the local truncations are truncations of y itself and the relative
  // error in the Frobenius norm stays below rankTolerance (as for a TT-SVD), also with many dimensions
  const std::vector<int> dims(8, 3);
  const TensorTrain TTx = positiveRandomTT<Type>(dims, std::vector<int>(7, 2), 21);
  const auto xDense = dense(TTx);
  const Eigen::Matrix<Type, Eigen::Dynamic, 1> fDense = xDense.array().sqrt();

  for(Type tol: {Type(1.e-2), Type(1.e-3)})
  {
    for(Type oversampling: {Type(0), Type(0.5)})
    {
      PITTS::TTCrossOptions<Type> options;
      options.rankTolerance = tol;
      options.oversampling = oversampling;
      options.nSweeps = 6;
      TensorTrain TTy(dims);
      applyElementwise([](Type v){return std::sqrt(v);}, TTx, TTy, options);
      EXPECT_LT((fDense - dense(TTy)).norm(), tol * fDense.norm());
    }
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, array_function)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  const std::vector<int> dims = {4, 3, 5, 3};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {2, 3, 2}, 5);

  // a function that is applied to all entries of a block at once must give the same result as the
  // corresponding scalar function (the algorithm only sees the same numbers); random error index sets
  // (randomErrorIndexSets) would require the same seed for both runs
  for(int nMALS: {1, 2})
  {
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;
    // Two runs are compared exactly, but the round-off differs between runs with threaded kernels (e.g. MKL). sqrt is
    // not exactly low-rank, so its singular values near sqrt(eps) are only as accurate as the approximation on the
    // crosses: rank decisions and the convergence check need tolerances clearly above that.
    options.rankTolerance = 100*eps;
    options.convergenceTolerance = 100*eps;

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
    EXPECT_NEAR(yRef, dense(TTy), 100*eps*yRef.norm());
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, convergence_measures)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  // mode dimensions and ranks vary strongly over the dimensions, so the blocks differ a lot in size:
  // the measures must still be comparable (rms scaling, one global reference)
  const std::vector<int> dims = {20, 2, 2, 2, 20};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {3, 2, 2, 3}, 23);

  TensorTrain TTy(dims);
  PITTS::TTCrossOptions<Type> options;
  // the ranks of two runs are compared below: x^2 has small genuine singular values (down to ~1e-6 relative), so the
  // truncation threshold must be clearly separated from them also in single precision (sqrt(eps) is not)
  options.rankTolerance = 10*eps;
  options.convergenceTolerance = 10*eps;
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
  EXPECT_NEAR(info.lastChange, infoScaled.lastChange, 1000*eps);
  EXPECT_NEAR(info.lastError, infoScaled.lastError, 1000*eps);

  // without enrichment there is no error estimate
  options.errorRank = 0;
  options.enrichmentRank = 0;
  TensorTrain TTnoEnrich(dims);
  const auto infoNoEnrich = applyElementwise([](Type v){return v*v;}, TTx, TTnoEnrich, options);
  EXPECT_EQ(Type(0), infoNoEnrich.lastError);

  // a run that is stopped while the ranks still grow is not converged (no comparison of runs: the default tolerances)
  options.rankTolerance = eps;
  options.convergenceTolerance = eps;
  options.errorRank = 3;
  options.enrichmentRank = 2;
  options.nSweeps = 1;
  TensorTrain TTshort(dims);
  const auto infoShort = applyElementwise([](Type v){return v*v;}, TTx, TTshort, options);
  // (in float the first sweep need not increase the ranks at the float tolerance: then only the measures decide,
  //  and they are at the level of the tolerance, so the result depends on round-off - only check the rank criterion)
  const auto maxRankOf = [](const std::vector<int>& ranks){return *std::max_element(ranks.begin(), ranks.end());};
  const bool ranksGrew = maxRankOf(TTshort.getTTranks()) > maxRankOf(TTx.getTTranks());
  if constexpr( std::is_same_v<Type, double> )
  {
    EXPECT_TRUE(ranksGrew);
  }
  EXPECT_FALSE(infoShort.converged && ranksGrew);
}

TYPED_TEST(PITTS_TensorTrain_cross, control_sample)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  const std::vector<int> dims = {4, 3, 5, 3};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {2, 3, 2}, 31);
  const auto f = [](Type v){return v*v;};

  PITTS::TTCrossOptions<Type> options;
  // the evaluations of two runs are compared below: truncation threshold clearly separated from the small genuine
  // singular values of x^2 also in single precision (see convergence_measures)
  options.rankTolerance = 10*eps;
  options.convergenceTolerance = 10*eps;
  options.nSweeps = 20;

  // same seed for both runs: also with random error index sets (randomErrorIndexSets) the sweeps are then
  // identical, so the difference in the number of evaluations is exactly the control sample
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
  EXPECT_LT(infoSampled.lastSampleError, 100*eps);
  // the control sample costs one function evaluation per sample (only once, the entries are fixed)
  EXPECT_EQ(info.nEvaluations + options.controlSampleSize, infoSampled.nEvaluations);

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
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

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
    options.rankTolerance = eps;
    options.nSweeps = 20;
    const auto info = applyElementwise([](Type v){return v*v;}, TTx, TTy, options);
    EXPECT_TRUE(info.converged);
    EXPECT_NEAR(ref, dense(TTy), 100*eps*ref.norm());
  }
}



TYPED_TEST(PITTS_TensorTrain_cross, small_dimensions)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  // one and two dimensions: the block of the first step is the whole tensor (nMALS is reduced to nDim),
  // no enrichment with a single core
  const auto f = [](Type v){return Type(1)/(Type(1)+v);};
  for(const std::vector<int>& dims: {std::vector<int>{7}, std::vector<int>{7, 5}})
    for(int nMALS: {1, 2})
    {
      const TensorTrain TTx = positiveRandomTT<Type>(dims, std::vector<int>(dims.size()-1, 2), 13);
      const Eigen::Matrix<Type, Eigen::Dynamic, 1> ref = dense(TTx).array().unaryExpr(f);
      PITTS::TTCrossOptions<Type> options;
      options.nMALS = nMALS;
      options.rankTolerance = Type(1.e-7);
      TensorTrain TTy(dims);
      const auto info = applyElementwise(f, TTx, TTy, options);
      EXPECT_TRUE(info.converged);
      EXPECT_NEAR(ref, dense(TTy), 100*eps*ref.norm());
    }
}

TYPED_TEST(PITTS_TensorTrain_cross, error_index_sets_with_low_rank_inputs)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  // The frame of rank-1 inputs only provides error crosses of size 1 < errorRank (randomErrorIndexSets = false),
  // the random ones have the full size: both must find the ranks of a function that is not low-rank, also for two
  // inputs (frame of their direct sum) and with oversampling.
  const std::vector<int> dims = {4, 4, 4, 4, 4};
  const TensorTrain TTx1 = positiveRandomTT<Type>(dims, {1, 1, 1, 1}, 17);
  const TensorTrain TTx2 = positiveRandomTT<Type>(dims, {1, 1, 1, 1}, 18);
  const auto x1Dense = dense(TTx1);
  const auto x2Dense = dense(TTx2);
  const Eigen::Matrix<Type, Eigen::Dynamic, 1> ref1 = (Type(1) + x1Dense.array()).inverse();
  const Eigen::Matrix<Type, Eigen::Dynamic, 1> ref2 = (Type(1) + x1Dense.array()*x2Dense.array()).inverse();

  for(bool random: {false, true})
    for(Type oversampling: {Type(0), Type(0.5)})
    {
      PITTS::TTCrossOptions<Type> options;
      options.randomErrorIndexSets = random;
      options.oversampling = oversampling;
      options.rankTolerance = Type(1.e-7);
      options.convergenceTolerance = Type(1.e-6);   // above the truncation error of rankTolerance, see TTCrossOptions
      options.nSweeps = 30;
      PITTS::internal::randomGenerator.seed(19);   // same random sets on all MPI processes

      TensorTrain TTy(dims);
      const auto info = applyElementwise([](Type v){return Type(1)/(Type(1)+v);}, TTx1, TTy, options);
      EXPECT_TRUE(info.converged);
      EXPECT_NEAR(ref1, dense(TTy), 100*eps*ref1.norm());
      EXPECT_GT(TTy.getTTranks()[1], 1);

      TensorTrain TTz(dims);
      const auto info2 = applyElementwise([](Type a, Type b){return Type(1)/(Type(1)+a*b);}, TTx1, TTx2, TTz, options);
      EXPECT_TRUE(info2.converged);
      EXPECT_NEAR(ref2, dense(TTz), 100*eps*ref2.norm());
      EXPECT_GT(TTz.getTTranks()[1], 1);
    }
}

// ---- functions of two tensor trains ----

TYPED_TEST(PITTS_TensorTrain_cross, two_inputs_invalid_arguments)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  const std::vector<int> dims = {3, 4, 2};
  TensorTrain TTx1(dims, 2), TTx2(dims, 3), TTwrong({3, 4, 3}, 2), TTy(dims);
  randomize(TTx1);
  randomize(TTx2);
  randomize(TTwrong);
  const auto mult = [](Type a, Type b){return a*b;};

  EXPECT_THROW(applyElementwise(mult, TTx1, TTwrong, TTy), std::invalid_argument);
  EXPECT_THROW(applyElementwise(mult, TTwrong, TTx2, TTy), std::invalid_argument);

  // boundary ranks != 1 of the second input
  std::vector<PITTS::Tensor3<Type>> cores(2);
  cores[0].resize(1, 3, 2);
  cores[1].resize(2, 4, 2);
  const std::vector<int> dims2 = {3, 4};
  TensorTrain TTboundary(std::move(cores)), TTx3(dims2, 2), TTy2(dims2);
  randomize(TTboundary);
  randomize(TTx3);
  EXPECT_THROW(applyElementwise(mult, TTx3, TTboundary, TTy2), std::invalid_argument);
}

TYPED_TEST(PITTS_TensorTrain_cross, two_inputs_product_and_quotient_vs_dense)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  // different ranks of the inputs
  const std::vector<int> dims = {4, 3, 5, 3};
  const TensorTrain TTx1 = positiveRandomTT<Type>(dims, {2, 3, 2}, 42);
  const TensorTrain TTx2 = positiveRandomTT<Type>(dims, {3, 1, 2}, 43);
  const auto x1Dense = dense(TTx1);
  const auto x2Dense = dense(TTx2);

  for(int nMALS: {1, 2})
  {
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;
    options.rankTolerance = eps;
    options.nSweeps = 20;

    // the element-wise (Hadamard) product has exact TT ranks <= r^(1) r^(2), so it is recovered exactly
    TensorTrain TTy(dims);
    const auto infoProduct = applyElementwise([](Type a, Type b){return a*b;}, TTx1, TTx2, TTy, options);
    EXPECT_TRUE(infoProduct.converged);
    const Eigen::Matrix<Type, Eigen::Dynamic, 1> productDense = x1Dense.array() * x2Dense.array();
    EXPECT_NEAR(productDense, dense(TTy), 100*eps*productDense.norm());
    for(int k = 0; k < dims.size()-1; k++)
      EXPECT_LE(TTy.getTTranks()[k], TTx1.getTTranks()[k]*TTx2.getTTranks()[k]);

    // the quotient is not exactly low-rank, but well approximable for entries in a moderate range;
    // f is not symmetric, so this also checks the order of the arguments
    TensorTrain TTz(dims);
    applyElementwise([](Type a, Type b){return a/b;}, TTx1, TTx2, TTz, options);
    const Eigen::Matrix<Type, Eigen::Dynamic, 1> quotientDense = x1Dense.array() / x2Dense.array();
    EXPECT_NEAR(quotientDense, dense(TTz), 10000*eps*quotientDense.norm());
  }
}

TYPED_TEST(PITTS_TensorTrain_cross, two_inputs_same_tensor_as_single_input)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  // f(x, x) = x*x with the same tensor twice: the stacked right unfoldings are rank-deficient, the initial
  // crosses have the size of those for a single input, and the result is x^2 as with the single-input variant.
  // (The crosses themselves may differ from the single-input ones: maxvol selects the columns in a different
  // basis of the same row space, so the number of sweeps can differ too.)
  const std::vector<int> dims = {4, 3, 5, 3};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {2, 3, 2}, 7);
  const auto xDense = dense(TTx);
  const Eigen::Matrix<Type, Eigen::Dynamic, 1> squareDense = xDense.array().square();

  PITTS::TTCrossOptions<Type> options;
  options.rankTolerance = eps;
  options.nSweeps = 20;
  TensorTrain TTy(dims);
  const auto info = applyElementwise([](Type a, Type b){return a*b;}, TTx, TTx, TTy, options);

  EXPECT_TRUE(info.converged);
  EXPECT_NEAR(squareDense, dense(TTy), 100*eps*squareDense.norm());
}

TYPED_TEST(PITTS_TensorTrain_cross, two_inputs_rank_growth)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  // the product of two random rank-3 tensors has rank 9 in the middle, the initial crosses only 3 + 3:
  // the enrichment must increase the ranks (general random cores: with positive ones the product is
  // dominated by a rank-1 part and well approximated with lower ranks)
  // (fixed seed: PITTS::initialize seeds each MPI process differently)
  const std::vector<int> dims = {4, 4, 4, 4, 4};
  TensorTrain TTx1(dims, 3), TTx2(dims, 3);
  PITTS::internal::randomGenerator.seed(4711);
  randomize(TTx1);
  randomize(TTx2);
  const Eigen::Matrix<Type, Eigen::Dynamic, 1> productDense = dense(TTx1).array() * dense(TTx2).array();

  PITTS::TTCrossOptions<Type> options;
  options.rankTolerance = eps;
  options.nSweeps = 20;
  TensorTrain TTy(dims);
  const auto info = applyElementwise([](Type a, Type b){return a*b;}, TTx1, TTx2, TTy, options);
  EXPECT_TRUE(info.converged);
  EXPECT_NEAR(productDense, dense(TTy), 100*eps*productDense.norm());
  EXPECT_GT(TTy.getTTranks()[1], 6);
  EXPECT_GT(TTy.getTTranks()[2], 6);
}

TYPED_TEST(PITTS_TensorTrain_cross, two_inputs_array_function)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  const std::vector<int> dims = {4, 3, 5, 3};
  const TensorTrain TTx1 = positiveRandomTT<Type>(dims, {2, 3, 2}, 5);
  const TensorTrain TTx2 = positiveRandomTT<Type>(dims, {2, 2, 2}, 6);

  // the array variant f(a, b) with the result in a must give the same result as the scalar function
  for(int nMALS: {1, 2})
  {
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;
    // Two runs are compared exactly, but the round-off differs between runs with threaded kernels (e.g. MKL). x1/x2 is
    // not exactly low-rank, so its singular values near sqrt(eps) are only as accurate as the approximation on the
    // crosses: rank decisions and the convergence check need tolerances clearly above that.
    options.rankTolerance = 100*eps;
    options.convergenceTolerance = 100*eps;

    TensorTrain TTy(dims), TTyRef(dims);
    PITTS::internal::randomGenerator.seed(2026);
    const auto info = applyElementwise([](PITTS::MultiVector<Type>& a, const PITTS::MultiVector<Type>& b)
    {
      for(long long i = 0; i < a.rows(); i++)
        a(i, 0) = a(i, 0) / b(i, 0);
    }, TTx1, TTx2, TTy, options);
    PITTS::internal::randomGenerator.seed(2026);
    const auto infoRef = applyElementwise([](Type a, Type b){return a/b;}, TTx1, TTx2, TTyRef, options);

    EXPECT_EQ(infoRef.nEvaluations, info.nEvaluations);
    EXPECT_EQ(infoRef.nSweeps, info.nSweeps);
    ASSERT_EQ(TTyRef.getTTranks(), TTy.getTTranks());
    const auto yRef = dense(TTyRef);
    EXPECT_NEAR(yRef, dense(TTy), 100*eps*yRef.norm());
  }
}


TYPED_TEST(PITTS_TensorTrain_cross, oversampling_invalid)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;

  const std::vector<int> dims = {3, 4, 2};
  TensorTrain TTx(dims, 2), TTy(dims);
  randomize(TTx);

  PITTS::TTCrossOptions<Type> options;
  options.oversampling = Type(-0.1);
  EXPECT_THROW(applyElementwise([](Type v){return v;}, TTx, TTy, options), std::invalid_argument);
}

TYPED_TEST(PITTS_TensorTrain_cross, oversampling_vs_dense)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  const std::vector<int> dims = {4, 3, 5, 3};
  const TensorTrain TTx = positiveRandomTT<Type>(dims, {2, 3, 2}, 42);
  const auto xDense = dense(TTx);

  for(int nMALS: {1, 2})
    for(Type oversampling: {Type(0.1), Type(1)})
    {
      PITTS::TTCrossOptions<Type> options;
      options.nMALS = nMALS;
      options.oversampling = oversampling;
      options.rankTolerance = eps;
      options.nSweeps = 20;

      // x^2 has exact TT ranks <= r^2: recovered exactly, and the ranks of y are the truncated ones
      // (not the sizes of the oversampled index sets)
      TensorTrain TTy(dims);
      const auto infoSquare = applyElementwise([](Type v){return v*v;}, TTx, TTy, options);
      EXPECT_TRUE(infoSquare.converged);
      const Eigen::Matrix<Type, Eigen::Dynamic, 1> squareDense = xDense.array().square();
      EXPECT_NEAR(squareDense, dense(TTy), 100*eps*squareDense.norm());
      for(int k = 0; k < dims.size()-1; k++)
        EXPECT_LE(TTy.getTTranks()[k], TTx.getTTranks()[k]*TTx.getTTranks()[k]);

      TensorTrain TTz(dims);
      applyElementwise([](Type v){return std::sqrt(v);}, TTx, TTz, options);
      const Eigen::Matrix<Type, Eigen::Dynamic, 1> sqrtDense = xDense.array().sqrt();
      EXPECT_NEAR(sqrtDense, dense(TTz), 10000*eps*sqrtDense.norm());
    }
}

TYPED_TEST(PITTS_TensorTrain_cross, oversampling_two_inputs_vs_dense)
{
  using Type = typename TestFixture::Type;
  using TensorTrain = PITTS::TensorTrain<Type>;
  const Type eps = std::sqrt(std::numeric_limits<Type>::epsilon());

  const std::vector<int> dims = {4, 3, 5, 3};
  const TensorTrain TTx1 = positiveRandomTT<Type>(dims, {2, 3, 2}, 42);
  const TensorTrain TTx2 = positiveRandomTT<Type>(dims, {3, 1, 2}, 43);
  const auto x1Dense = dense(TTx1);
  const auto x2Dense = dense(TTx2);

  for(int nMALS: {1, 2})
  {
    PITTS::TTCrossOptions<Type> options;
    options.nMALS = nMALS;
    options.oversampling = Type(0.3);
    options.rankTolerance = eps;
    options.nSweeps = 20;

    TensorTrain TTy(dims);
    const auto info = applyElementwise([](Type a, Type b){return a*b;}, TTx1, TTx2, TTy, options);
    EXPECT_TRUE(info.converged);
    const Eigen::Matrix<Type, Eigen::Dynamic, 1> productDense = x1Dense.array() * x2Dense.array();
    EXPECT_NEAR(productDense, dense(TTy), 100*eps*productDense.norm());
  }
}

namespace
{
  // x = Phi Phi^T for unit-norm random features Phi (N x R) as a TT with the layout (i_1, ..., i_k, j_k, ..., j_1),
  // the digits of the row index i and the column index j (first digit fastest) are unstructured.
  // The kernel exp(-2 gamma (1 - x)) = exp(-gamma |phi_i - phi_j|^2) of such a Gram matrix has high ranks in the TT,
  // far above the ranks the first sweeps can reach with enrichment: an under-resolved approximation.
  PITTS::TensorTrain<double> randomGramTensorTrain(const std::vector<int>& digits, int R, unsigned seed)
  {
    const int k = digits.size();
    long long N = 1;
    for(int n: digits)
      N *= n;

    std::mt19937 gen(seed);
    std::normal_distribution<double> dist;
    Eigen::MatrixXd Phi(N, R);
    for(long long i = 0; i < N; i++)
      for(int r = 0; r < R; r++)
        Phi(i, r) = dist(gen);
    Phi.rowwise().normalize();
    const Eigen::MatrixXd gram = Phi * Phi.transpose();

    std::vector<int> dims = digits;
    dims.insert(dims.end(), digits.rbegin(), digits.rend());
    std::vector<long long> stride(2*k, 1);
    for(int m = 1; m < 2*k; m++)
      stride[m] = stride[m-1] * dims[m-1];

    std::vector<double> data(N*N);
    for(long long I = 0; I < N; I++)
      for(long long J = 0; J < N; J++)
      {
        long long position = 0, rest = I;
        for(int m = 0; m < k; m++)
        {
          position += (rest % digits[m]) * stride[m];
          rest /= digits[m];
        }
        rest = J;
        for(int m = 0; m < k; m++)
        {
          position += (rest % digits[m]) * stride[2*k-1-m];
          rest /= digits[m];
        }
        data[position] = gram(I, J);
      }
    return PITTS::fromDense_classical(data.cbegin(), data.cend(), dims, 1.e-14);
  }
}

TEST(PITTS_TensorTrain_cross_robustness, oversampling_for_under_resolved_ranks)
{
  using TensorTrain = PITTS::TensorTrain<double>;

  // small enough for debug builds (N = 72 points, x is 72 x 72), yet the ranks are far above those of the first sweeps
  const TensorTrain TTx = randomGramTensorTrain({2, 2, 2, 3, 3}, 4, 17);
  const auto xDense = dense(TTx);
  const double gamma = 1;
  const auto kernel = [gamma](double v){return std::exp(-2*gamma*(1-v));};
  const Eigen::VectorXd yRef = xDense.unaryExpr(kernel);

  // the error of y with respect to the exact kernel (all entries) after a given number of sweeps
  const auto errorAfter = [&](int nSweeps, double oversampling)
  {
    PITTS::TTCrossOptions<double> options;
    options.nSweeps = nSweeps;
    options.oversampling = oversampling;
    options.enrichmentRank = 4;
    options.errorRank = 10;
    PITTS::internal::randomGenerator.seed(2026);
    TensorTrain TTy(TTx.dimensions());
    applyElementwise(kernel, TTx, TTy, options);
    return (dense(TTy) - yRef).norm() / yRef.norm();
  };

  // The ranks grow by the enrichment in both variants. After 8 sweeps the square (collocation) interpolation
  // reaches errors of 0.07-0.12, the least-squares interpolation 0.010-0.014 (observed for several random features;
  // in larger cases the square variant even gets worse with the number of sweeps).
  const double errorFirstSweep = errorAfter(1, 0.5);
  EXPECT_LT(errorFirstSweep, 1);
  EXPECT_LT(errorAfter(8, 0.5), 0.03);
}
