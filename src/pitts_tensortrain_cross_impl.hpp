// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_cross_impl.hpp
* @brief TT-cross approximation: apply a function element-wise to a tensor in TT format
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-11
*
* Sweep driver on top of the building blocks in pitts_tensortrain_cross_helper.hpp. Templated on the
* function type, so this file is always included by pitts_tensortrain_cross.hpp (also in the development
* build); the heavy parts (contractions, SVD, maxvol) are the explicitly instantiated helpers.
* The driver is written for a function of N tensor trains: everything that depends on the inputs is
* encapsulated in internal::tt_cross::FunctionOnCrosses, the public functions only select N.
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_CROSS_IMPL_HPP
#define PITTS_TENSORTRAIN_CROSS_IMPL_HPP

// includes
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include "pitts_tensortrain_cross.hpp"
#include "pitts_tensortrain_cross_helper.hpp"
#include "pitts_tensortrain_axpby_normalized.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_sweep_data.hpp"
#include "pitts_tensortrain_sweep_index.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! dedicated helper functionality for TT-cross algorithms
  namespace internal::tt_cross
  {
  //! TT-cross sweeps for TTy ~ f(TTx[0], ..., TTx[N-1]) element-wise, see applyElementwise
  template<typename T, typename Function, std::size_t N>
  TTCrossInfo<T> crossSweeps(const Function& f, const std::array<std::reference_wrapper<const TensorTrain<T>>, N>& TTx,
                             TensorTrain<T>& TTy, const TTCrossOptions<T>& options)
  {
    using RealType = decltype(std::abs(T()));

    // check arguments
    const std::vector<int>& dims = TTy.dimensions();
    const int nDim = dims.size();
    if( nDim < 1 )
      throw std::invalid_argument("applyElementwise: empty tensor train!");
    for(const TensorTrain<T>& TT: TTx)
    {
      if( TT.dimensions() != dims )
        throw std::invalid_argument("applyElementwise: dimensions of TTx and TTy differ!");
      if( TT.subTensor(0).r1() != 1 || TT.subTensor(nDim-1).r2() != 1 )
        throw std::invalid_argument("applyElementwise: boundary ranks of TTx must be 1!");
    }
    if( options.nMALS < 1 || options.nMALS > 2 )
      throw std::invalid_argument("applyElementwise: nMALS must be 1 or 2!");
    if( options.errorRank < 0 || options.enrichmentRank < 0 || options.enrichmentRank > options.errorRank )
      throw std::invalid_argument("applyElementwise: need 0 <= enrichmentRank <= errorRank!");
    if( options.nSweeps < 1 || options.maxRank < 1 )
      throw std::invalid_argument("applyElementwise: nSweeps and maxRank must be positive!");
    if( !options.maxRanks.empty() && (std::ssize(options.maxRanks) != nDim-1 || *std::min_element(options.maxRanks.begin(), options.maxRanks.end()) < 1) )
      throw std::invalid_argument("applyElementwise: maxRanks needs one positive entry per bond!");
    if( !(std::real(options.oversampling) >= 0) )
      throw std::invalid_argument("applyElementwise: oversampling must not be negative!");

    const int nMALS = std::min(options.nMALS, nDim);
    // enrichment only for the 1-site variant (the 2-site variant adapts the ranks through the SVD)
    const bool enrich = nMALS == 1 && nDim > 1 && options.errorRank > 0 && options.enrichmentRank > 0;
    const T localTolerance = options.rankTolerance / (2*std::sqrt(T(std::max(nDim-1, 1))));     // factor 2 heuristic, like tt_amen in ttpy
    const T maxvolTolerance = options.maxvolTolerance;
    // maximal rank per bond k (between cores k and k+1)
    const std::vector<int> maxRanks = options.maxRanks.empty() ? std::vector<int>(std::max(nDim-1, 0), options.maxRank) : options.maxRanks;
    const T oversampling = options.oversampling;
    const bool oversample = std::real(oversampling) > 0;

    TTCrossInfo<T> info;

    // index sets of y: I[k] for k = 0..nDim-2, J[k] for k = 1..nDim-1 (J[0] unused)
    std::vector<NestedIndices> I(nDim);
    std::vector<NestedIndices> J = initialRightIndexSets(std::vector<std::reference_wrapper<const TensorTrain<T>>>(TTx.begin(), TTx.end()), maxvolTolerance);

    // index sets of the error approximation z (no cores needed, only the crosses)
    std::vector<NestedIndices> Iz(nDim), Jz(nDim);
    if( enrich )
    {
      if( options.randomErrorIndexSets )
        Jz = randomRightIndexSets(dims, options.errorRank);
      else
      {
        // maxvol sets of a rank-errorRank truncation of x: well-conditioned crosses of the dominant part of x
        std::vector<TensorTrain<T>> TTxTrunc;
        TTxTrunc.reserve(N);
        for(const TensorTrain<T>& TT: TTx)
        {
          copy(TT, TTxTrunc.emplace_back(dims));
          normalize(TTxTrunc.back(), T(0), options.errorRank);
        }
        Jz = initialRightIndexSets(std::vector<std::reference_wrapper<const TensorTrain<T>>>(TTxTrunc.begin(), TTxTrunc.end()), maxvolTolerance, options.errorRank);
      }
    }

    // f(x) on the crosses of y and z (with the partial contractions of x on them), partial contractions of y on the crosses of z
    FunctionOnCrosses<T, Function, N> fx(f, TTx, I, J, Iz, Jz);
    auto Lyz = defineSweepData<Tensor3<T>>(nDim, contraction_loop_from_left<T>(TTy, Iz), contraction_loop_from_right<T>(TTy, Jz));
    // partial contractions of y on its own crosses; only needed with oversampling, see yOnOwnCross
    auto Ly = defineSweepData<Tensor3<T>>(nDim, contraction_loop_from_left<T>(TTy, I), contraction_loop_from_right<T>(TTy, J));

    // y on the cross (I_{k-1}, S_k x ... x S_{k+nMALS-1}, J_{k+nMALS}) of the block starting at core k.
    // Without oversampling y is in the interpolation gauge: the selected rows (columns) of the cores on the left
    // (right) are the identity, so y on its own cross is just the (combined) core. With oversampling the cores only
    // reproduce f(x) in the least-squares sense and both interfaces have to be contracted.
    const auto yOnOwnCross = [&](int k) -> Tensor3<T>
    {
      Tensor3<T> core = combineCores(TTy, k, nMALS);
      if( !oversample )
        return core;
      Ly.update(k-1, k+nMALS);
      Tensor3<T> withLeft, result;
      contractLeftInterface<T>(Ly.left(), I, k-1, core, withLeft);
      contractRightInterface<T>(withLeft, Ly.right(), J, k+nMALS, result);
      return result;
    };

    // Start with a zero y of the initial cross ranks (|J_{k+1}| at bond k): a sweep replaces the cores one
    // by one and multiplies the remainder of each local update into the next core, so y must be a tensor
    // train of exactly these ranks before the first step. In the later sweeps this holds automatically
    // because the opposite sweep selects the index set and the rank of a core together (see stepRightToLeft).
    // Note: the first left-to-right sweep then contracts and subtracts these zeros in the enrichment and in
    // the convergence measure. That work could be optimized away, but it only pays off if the iteration
    // converges within the very first sweeps.
    TTy.setSubTensors(0, [&]()
    {
      std::vector<Tensor3<T>> zeroCores(nDim);
      for(int k = 0; k < nDim; k++)
      {
        zeroCores[k].resize(k == 0 ? 1 : J[k].size(), dims[k], k+1 == nDim ? 1 : J[k+1].size());
        zeroCores[k].setConstant(T(0));
      }
      return zeroCores;
    }());

    // With oversampling (1-site), the extra rows/columns of a cross would also count as rank in the next block,
    // so the ranks would grow by a factor (1+oversampling)^2 per sweep (whenever f(x) is not resolved yet).
    // Of the possible ways to limit this, the growth is left to the enrichment here: a left-to-right step keeps
    // at most the truncated rank of the bond from the last right-to-left sweep (new directions only from the
    // enrichment), a right-to-left step at most that rank + enrichmentRank (the dominant directions of the block).
    std::vector<int> bondRank(nDim);
    for(int k = 0; k+1 < nDim; k++)
      bondRank[k] = J[k+1].size();
    const bool limitGrowth = oversample && nMALS == 1;

    // Convergence measures of a step, both returned separately: the change of y on its own cross (block
    // vs. current core) and, with enrichment, the error f(x) - y on the error crosses. The change alone is
    // blind when the crosses of y are too small but y is exact on them (e.g. a rank that is too low at some
    // bond); the error on the (different) error crosses detects this and prevents an early stop.
    //
    // Both are root-mean-square sizes (t3_rms) divided by rmsRef, the largest block rms seen so far:
    //  * rms instead of the Frobenius norm, because the crosses grow with the TT ranks and the error crosses
    //    have a different (fixed) size - Frobenius norms of differently sized blocks are not comparable;
    //  * one global reference instead of the norm of each block, because a bond whose cross samples a region
    //    with small values would otherwise dominate the maximum over the sweep with a large relative value.
    RealType rmsRef = 0;

    // ---- one step of a left-to-right sweep at the cores swpIdx.leftDim()..swpIdx.rightDim() ----
    // returns the convergence measures (change, error) of the step
    const auto stepLeftToRight = [&](const SweepIndex& swpIdx) -> std::pair<RealType, RealType>
    {
      const int k = swpIdx.leftDim();
      const int kRight = swpIdx.rightDim();

      // block B = f( x(I_{k-1}, S_k [x S_{k+1}], J_{kRight+1}) ) = f( LX_k [X_{k+1}] R_{kRight+1} );
      // LX_k = L_{k-1} X_k is cached: it is reused for the enrichment residual and for the next interface L_k
      Tensor3<T> B = fx.blockFromLeft(k, nMALS);
      rmsRef = std::max(rmsRef, t3_rms(B));
      const RealType scale = rmsRef > 0 ? rmsRef : 1;   // zero blocks only: the measures become absolute

      // convergence measure: the current y on the same cross (the current core(s) without oversampling)
      RealType change = rmsDifference(B, yOnOwnCross(k)) / scale;
      RealType error = 0;

      // last core of the 1-site sweep: the block is the core, nothing to select
      if( nMALS == 1 && k == nDim-1 )
      {
        TTy.setSubTensor(k, std::move(B));
        Lyz.invalidate(k);   // the core of y changed
        Ly.invalidate(k);
        return {change, error};
      }

      // enrichment: dominant directions of the error f(x) - y on the cross (I_{k-1}, S_k, J^z_{k+1}),
      // i.e. of f(LX_k R^z_{k+1}) - Y_k R^{yz}_{k+1}
      Tensor3<T> E;   // empty unless this step enriches
      if( enrich )
      {
        Tensor3<T> residual = fx.mixedBlock(k);
        // y on the cross: the left interface of y at I_{k-1} is the identity (interpolation gauge), so only the right part remains
        Lyz.update(k-1, k+1);
        Tensor3<T> yOnMixedCross;
        if( oversample )
        {
          // the left interface of y at I_{k-1} is not the identity
          Ly.update(k-1, k+nMALS);
          Tensor3<T> withLeft;
          contractLeftInterface<T>(Ly.left(), I, k-1, TTy.subTensor(k), withLeft);
          contractRightInterface<T>(withLeft, Lyz.right(), Jz, k+1, yOnMixedCross);
        }
        else
          contractRightInterface<T>(TTy.subTensor(k), Lyz.right(), Jz, k+1, yOnMixedCross);
        internal::t3_axpy(T(-1), yOnMixedCross, residual);
        error = std::max(error, t3_rms(residual) / scale);
        // as in solveMALS, updateCoreLeft first projects out the current directions and then appends the
        // dominant (at most enrichmentRank) new directions of the error
        E = std::move(residual);
      }

      // local update: truncated SVD, enrichment, maxvol, interpolation gauge
      // (2-site: view the block r1 x n_k n_{k+1} x r2 as r1 x n_k x n_{k+1} r2, same memory layout, so that
      //  the left unfolding of the update splits off core k)
      if( nMALS == 2 )
        B = fold_right(unfold_right(std::move(B)), dims[k]);
      const int rPrev = B.r1();
      const int maxRankLeft = limitGrowth ? std::min(maxRanks[k], bondRank[k]) : maxRanks[k];
      CoreUpdate<T> upd = updateCoreLeft<T>(B, localTolerance, maxRankLeft, maxvolTolerance, std::move(E), oversampling, options.enrichmentRank);
      I[k] = leftIndicesFromRows(upd.selected, rPrev);
      fx.invalidate(k+1);

      // new core(s); the carry keeps y consistent as a tensor until the next core is rebuilt
      std::vector<Tensor3<T>> newCores(2);
      newCores[0] = std::move(upd.core);
      if( nMALS == 2 )
        newCores[1] = fold_right(std::move(upd.carry), dims[k+1]);
      else
        internal::normalize_contract1(upd.carry, TTy.subTensor(k+1), newCores[1]);
      TTy.setSubTensors(k, std::move(newCores));
      Ly.invalidate(k, 2);

      // update of the error crosses: I^z_k from the error f(x) - y on (I^z_{k-1}, S_k, J^z_{k+1}),
      // i.e. of f(LX^z_k R^z_{k+1}) - LY^z_k R^{yz}_{k+1}
      if( enrich )
      {
        Lyz.invalidate(k, 2);
        Tensor3<T> residual = fx.errorBlockFromLeft(k);
        Lyz.update(k, k+1);
        Tensor3<T> yOnCross;
        contractRightInterface<T>(Lyz.left()->get(), Lyz.right(), Jz, k+1, yOnCross);
        internal::t3_axpy(T(-1), yOnCross, residual);
        error = std::max(error, t3_rms(residual) / scale);
        const int rPrevZ = residual.r1();
        const CoreUpdate<T> updz = updateCoreLeft<T>(residual, localTolerance, options.errorRank, maxvolTolerance);
        Iz[k] = leftIndicesFromRows(updz.selected, rPrevZ);
        fx.invalidateError(k+1);
        Lyz.invalidate(k+1);
      }

      return {change, error};
    };

    // ---- one step of a right-to-left sweep at the cores swpIdx.leftDim()..swpIdx.rightDim() (mirror image) ----
    const auto stepRightToLeft = [&](const SweepIndex& swpIdx) -> std::pair<RealType, RealType>
    {
      const int k = swpIdx.rightDim();
      const int kLeft = swpIdx.leftDim();

      // block B = f( x(I_{kLeft-1}, [S_{k-1} x] S_k, J_{k+1}) ) = f( L_{kLeft-1} [X_{k-1}] XR_k ) with the cached XR_k = X_k R_{k+1}
      Tensor3<T> B = fx.blockFromRight(kLeft, nMALS);
      rmsRef = std::max(rmsRef, t3_rms(B));
      const RealType scale = rmsRef > 0 ? rmsRef : 1;

      RealType change = rmsDifference(B, yOnOwnCross(kLeft)) / scale;
      RealType error = 0;

      // first core of the 1-site sweep: the block is the core
      if( nMALS == 1 && k == 0 )
      {
        TTy.setSubTensor(k, std::move(B));
        Lyz.invalidate(k);
        Ly.invalidate(k);
        return {change, error};
      }

      // No enrichment in this direction: the right-to-left sweep only truncates, so that after each full
      // sweep the ranks of y are the truncated ranks (not rank + enrichmentRank as in amen_cross, which
      // leaves the final truncation to the caller). Rank growth still happens: the enlarged left sets
      // I_{k-1} of the preceding left-to-right sweep provide the additional rows of the block here.
      // (2-site: view the block r1 x n_{k-1} n_k x r2 as r1 n_{k-1} x n_k x r2 for splitting off core k)
      if( nMALS == 2 )
        B = fold_left(unfold_left(std::move(B)), dims[k]);
      const int maxRankRight = limitGrowth ? std::min(maxRanks[k-1], bondRank[k-1] + options.enrichmentRank) : maxRanks[k-1];
      CoreUpdate<T> upd = updateCoreRight<T>(B, localTolerance, maxRankRight, maxvolTolerance, oversampling);
      bondRank[k-1] = upd.truncatedRank;
      J[k] = rightIndicesFromColumns(upd.selected, dims[k]);
      fx.invalidate(k-1);

      std::vector<Tensor3<T>> newCores(2);
      if( nMALS == 2 )
        newCores[0] = fold_left(std::move(upd.carry), dims[k-1]);
      else
        internal::normalize_contract2(TTy.subTensor(k-1), upd.carry, newCores[0]);
      newCores[1] = std::move(upd.core);
      TTy.setSubTensors(k-1, std::move(newCores));
      Ly.invalidate(k-1, 2);

      // update of the error crosses: J^z_k from the error f(x) - y on (I^z_{k-1}, S_k, J^z_{k+1}),
      // i.e. of f(L^z_{k-1} XR^z_k) - L^{yz}_{k-1} YR^z_k
      if( enrich )
      {
        Lyz.invalidate(k-1, 2);
        Tensor3<T> residual = fx.errorBlockFromRight(k);
        Lyz.update(k-1, k);
        Tensor3<T> yOnCross;
        contractLeftInterface<T>(Lyz.left(), Iz, k-1, Lyz.right()->get(), yOnCross);
        internal::t3_axpy(T(-1), yOnCross, residual);
        error = std::max(error, t3_rms(residual) / scale);
        const CoreUpdate<T> updz = updateCoreRight<T>(residual, localTolerance, options.errorRank, maxvolTolerance);
        Jz[k] = rightIndicesFromColumns(updz.selected, dims[k]);
        fx.invalidateError(k-1);
        Lyz.invalidate(k-1);
      }

      return {change, error};
    };

    // ---- sweeps ----
    // independent control sample: random entries, drawn once so that the estimate is comparable between
    // sweeps; it does not enter the convergence check, see TTCrossOptions::controlSampleSize
    const Eigen::MatrixXi sampleIndices = options.controlSampleSize > 0 ?
        randomIndices(dims, options.controlSampleSize) : Eigen::MatrixXi(0, nDim);

    // f(x) on the sample does not change (fixed entries), so it is evaluated only once; y after every half sweep
    const Tensor3<T> fOnSample = sampleIndices.rows() > 0 ? fx.atIndices(sampleIndices) : Tensor3<T>();
    const RealType rmsfOnSample = sampleIndices.rows() > 0 ? t3_rms(fOnSample) : RealType(0);
    const auto sampleError = [&]() -> RealType
    {
      if( sampleIndices.rows() == 0 )
        return 0;
      Tensor3<T> yOnSample;
      evaluateAt(TTy, sampleIndices, yOnSample);
      return rmsDifference(fOnSample, yOnSample) / (rmsfOnSample > 0 ? rmsfOnSample : 1);
    };

    const auto printProgress = [&](double sweep, RealType change, RealType error)
    {
      std::cout << "TT-cross sweep " << sweep << ": change " << change << ", error estimate " << error;
      if( options.controlSampleSize > 0 )
        std::cout << ", sampled error " << info.lastSampleError;
      std::cout << ", ranks:";
      for(int r: TTy.getTTranks())
        std::cout << " " << r;
      std::cout << "\n";
    };

    const SweepIndex firstIdx(nDim, nMALS, nMALS-1, 0);
    // Rank growth as a second convergence criterion: only the left-to-right sweep enriches, the
    // right-to-left sweep truncates, so the ranks after a full sweep are always the truncated ones and
    // comparing them between full sweeps tells whether the approximation is still growing. The maximal
    // rank is used rather than each individual one: single bonds may still change by +-1 near convergence.
    const auto maxRankOf = [](const std::vector<int>& ranks){return ranks.empty() ? 0 : *std::max_element(ranks.begin(), ranks.end());};
    int largestRank = maxRankOf(TTy.getTTranks());
    for(int iSweep = 0; iSweep < options.nSweeps; iSweep++)
    {
      // left to right (the very first one builds y from scratch, its change measure is meaningless)
      RealType maxChange = 0, maxError = 0;
      for(SweepIndex swpIdx = firstIdx.first(); swpIdx; swpIdx = swpIdx.next())
      {
        const auto [change, error] = stepLeftToRight(swpIdx);
        maxChange = std::max(maxChange, change);
        maxError = std::max(maxError, error);
      }
      info.lastSampleError = sampleError();
      if( options.verbose )
        printProgress(iSweep+0.5, maxChange, maxError);

      // right to left (this also truncates the enrichment of the left-to-right sweep, so the convergence
      // check only makes sense on the ranks after the full sweep)
      maxChange = maxError = 0;
      for(SweepIndex swpIdx = firstIdx.last(); swpIdx; swpIdx = swpIdx.previous())
      {
        const auto [change, error] = stepRightToLeft(swpIdx);
        maxChange = std::max(maxChange, change);
        maxError = std::max(maxError, error);
      }
      info.nSweeps = iSweep+1;
      info.lastChange = maxChange;
      info.lastError = maxError;
      info.lastSampleError = sampleError();
      if( options.verbose )
        printProgress(iSweep+1, maxChange, maxError);

      // a rank that still grows means the algorithm itself is not done, whatever the measures say
      const int previousRank = std::exchange(largestRank, maxRankOf(TTy.getTTranks()));
      if( maxChange < options.convergenceTolerance && maxError < options.convergenceTolerance && largestRank <= previousRank )
      {
        info.converged = true;
        break;
      }
    }

    // with oversampling the bonds of y have the sizes of the index sets, which exceed the ranks:
    // truncate (normalize also scales y to norm 1 and returns the norm, which is put back into the core with the norm).
    // The largest per-bond limit suffices: the actual rank of y at bond k is already at most maxRanks[k] (after the
    // last right-to-left sweep, the right unfolding of core k+1 has the truncated rank), only the bond size is larger.
    if( oversample && nDim > 1 )
    {
      const T nrm = normalize(TTy, options.rankTolerance, *std::max_element(maxRanks.begin(), maxRanks.end()));
      const int idxNorm = (TTy.isOrthogonal() & TT_Orthogonality::left) != TT_Orthogonality::none ? nDim-1 : 0;
      TTy.editSubTensor(idxNorm, [nrm](Tensor3<T>& subT){internal::t3_scale(nrm, subT);});
    }

    info.nEvaluations = fx.nEvaluations();

    // final index sets as full multi-indices
    info.leftIndices.resize(std::max(nDim-1, 0));
    for(int k = 0; k < nDim-1; k++)
      info.leftIndices[k] = expandLeft(I, k);
    info.rightIndices.resize(nDim);
    for(int k = 1; k < nDim; k++)
      info.rightIndices[k] = expandRight(J, k, nDim);

    return info;
  }
  }


  // implement applyElementwise
  template<typename T, typename Function>
  TTCrossInfo<T> applyElementwise(const Function& f, const TensorTrain<T>& TTx, TensorTrain<T>& TTy, const TTCrossOptions<T>& options)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    return internal::tt_cross::crossSweeps<T, Function, 1>(f, {std::cref(TTx)}, TTy, options);
  }

  // implement applyElementwise for two inputs
  template<typename T, typename Function>
  TTCrossInfo<T> applyElementwise(const Function& f, const TensorTrain<T>& TTx1, const TensorTrain<T>& TTx2, TensorTrain<T>& TTy, const TTCrossOptions<T>& options)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    return internal::tt_cross::crossSweeps<T, Function, 2>(f, {std::cref(TTx1), std::cref(TTx2)}, TTy, options);
  }
}


#endif // PITTS_TENSORTRAIN_CROSS_IMPL_HPP
