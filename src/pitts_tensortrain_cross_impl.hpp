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
    // Limits per bond k (between cores k and k+1): a TT rank at bond k is at most n_k times the rank at bond k-1 and
    // n_{k+1} times the rank at bond k+1, so a small bond also limits its neighbors (one pass in each direction suffices).
    const auto limitByNeighbors = [&dims, nDim](std::vector<long long> limit)
    {
      for(int k = 0; k < nDim-1; k++)
        limit[k] = std::min(limit[k], dims[k] * (k == 0 ? 1 : limit[k-1]));
      for(int k = nDim-2; k >= 0; k--)
        limit[k] = std::min(limit[k], dims[k+1] * (k == nDim-2 ? 1 : limit[k+1]));
      return std::vector<int>(limit.begin(), limit.end());
    };
    // maximal useful rank per bond (also after the enrichment)
    const std::vector<int> maxRanks = limitByNeighbors(options.maxRanks.empty() ?
      std::vector<long long>(std::max(nDim-1, 0), options.maxRank) : std::vector<long long>(options.maxRanks.begin(), options.maxRanks.end()));
    // maximal size of the index sets per bond: the oversampled maximal rank, again limited by the neighbors
    std::vector<long long> oversampledRanks(maxRanks.begin(), maxRanks.end());
    for(long long& q: oversampledRanks)
      q += static_cast<long long>(std::ceil(std::real(options.oversampling) * q));
    const std::vector<int> maxIndices = limitByNeighbors(std::move(oversampledRanks));
    const T oversampling = options.oversampling;

    TTCrossInfo<T> info;

    // y is kept in the orthogonal gauge, like the solution in solveMALS: in a sweep, the cores left of the current
    // block are left-orthonormal and the cores right of it right-orthonormal, so the truncated SVD of a local block
    // is the truncation of y itself. Start: y = 0 with an orthonormal right frame spanned by the inputs.
    const std::vector<std::reference_wrapper<const TensorTrain<T>>> inputs(TTx.begin(), TTx.end());
    initialRightFrame(inputs, TTy);

    // index sets of y: I[k] for k = 0..nDim-2, J[k] for k = 1..nDim-1 (J[0] unused);
    // initial right index sets from a maxvol sweep over the right frame
    std::vector<NestedIndices> I(nDim);
    std::vector<NestedIndices> J = rightIndexSetsFromFrame(TTy, maxvolTolerance, oversampling, maxIndices);

    // partial contractions of y on its own crosses: their row/column selections are the interface matrices of y
    // (the values of the orthonormal left/right parts on I_k resp. J_k), which convert f(x) on a cross into the
    // coordinates of y (least squares with oversampling) and y back onto the cross
    auto Ly = defineSweepData<Tensor3<T>>(nDim, contraction_loop_from_left<T>(TTy, I), contraction_loop_from_right<T>(TTy, J));

    // index sets of the error approximation z (no cores needed, only the crosses)
    std::vector<NestedIndices> Iz(nDim), Jz(nDim);
    if( enrich )
    {
      if( options.randomErrorIndexSets )
        Jz = randomRightIndexSets(dims, options.errorRank);
      else
      {
        // maxvol sets of a frame of rank errorRank of the inputs: well-conditioned crosses of the dominant part of x
        TensorTrain<T> TTz(dims);
        initialRightFrame(inputs, TTz, options.errorRank);
        Jz = rightIndexSetsFromFrame(TTz, maxvolTolerance);
      }
    }

    // f(x) on the crosses of y and z (with the partial contractions of x on them), partial contractions of y on the crosses of z
    FunctionOnCrosses<T, Function, N> fx(f, TTx, I, J, Iz, Jz);
    auto Lyz = defineSweepData<Tensor3<T>>(nDim, contraction_loop_from_left<T>(TTy, Iz), contraction_loop_from_right<T>(TTy, Jz));

    // For the block of the cores k..k+nMALS-1 (all three need Ly.update(k-1, k+nMALS) first):
    // y on its cross (I_{k-1}, S_k x ... x S_{k+nMALS-1}, J_{k+nMALS})
    const auto yOnOwnCross = [&](int k)
    {
      Tensor3<T> withLeft, result;
      contractLeftInterface<T>(Ly.left(), I, k-1, combineCores(TTy, k, nMALS), withLeft);
      contractRightInterface<T>(withLeft, Ly.right(), J, k+nMALS, result);
      return result;
    };
    // a block on this cross in the coordinates of y
    const auto toCoordinatesOfY = [&](int k, const Tensor3<T>& B)
    {
      Tensor3<T> withLeft, result;
      solveLeftInterface<T>(Ly.left(), I, k-1, B, withLeft);
      solveRightInterface<T>(withLeft, Ly.right(), J, k+nMALS, result);
      return result;
    };
    // the left resp. right interface matrix of y at this cross (the 1x1 identity at the boundary)
    const auto identity1 = []{Tensor2<T> one(1, 1); one(0, 0) = T(1); return one;};
    const auto leftInterfaceOfY = [&](int k){return Ly.left() ? interfaceLeft(Ly.left()->get(), I[k-1]) : identity1();};
    const auto rightInterfaceOfY = [&](int k){return Ly.right() ? interfaceRight(Ly.right()->get(), J[k+nMALS]) : identity1();};

    // Convergence measures of a step, both returned separately: the change of y on its own cross (block
    // vs. current y) and, with enrichment, the error f(x) - y on the error crosses. The change alone is
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

      // block B = f( x(I_{k-1}, S_k [x S_{k+1}], J_{kRight+1}) ) = f( LX_k [X_{k+1}] R_{kRight+1} );
      // LX_k = L_{k-1} X_k is cached: it is reused for the enrichment residual and for the next interface L_k
      const Tensor3<T> B = fx.blockFromLeft(k, nMALS);
      rmsRef = std::max(rmsRef, t3_rms(B));
      const RealType scale = rmsRef > 0 ? rmsRef : 1;   // zero blocks only: the measures become absolute

      // convergence measure: the current y on the same cross
      Ly.update(k-1, k+nMALS);
      RealType change = rmsDifference(B, yOnOwnCross(k)) / scale;
      RealType error = 0;

      Tensor3<T> C = toCoordinatesOfY(k, B);

      // last core of the 1-site sweep: the block is the core, nothing to select
      if( nMALS == 1 && k == nDim-1 )
      {
        TTy.setSubTensor(k, std::move(C));
        Lyz.invalidate(k);   // the core of y changed
        Ly.invalidate(k);
        return {change, error};
      }

      // enrichment: dominant directions of the error f(x) - y on the cross (I_{k-1}, S_k, J^z_{k+1}),
      // i.e. of f(LX_k R^z_{k+1}) - LY_k R^{yz}_{k+1}, in the left coordinates of y
      Tensor3<T> E;   // empty unless this step enriches
      if( enrich )
      {
        Tensor3<T> residual = fx.mixedBlock(k);
        Lyz.update(k-1, k+1);
        Tensor3<T> withLeft, yOnMixedCross;
        contractLeftInterface<T>(Ly.left(), I, k-1, TTy.subTensor(k), withLeft);
        contractRightInterface<T>(withLeft, Lyz.right(), Jz, k+1, yOnMixedCross);
        internal::t3_axpy(T(-1), yOnMixedCross, residual);
        error = std::max(error, t3_rms(residual) / scale);
        // as in solveMALS, updateCoreLeft first projects out the current directions and then appends the
        // dominant (at most enrichmentRank) new directions of the error
        solveLeftInterface<T>(Ly.left(), I, k-1, residual, E);
      }

      // local update: truncated SVD, enrichment, maxvol
      // (2-site: view the block r1 x n_k n_{k+1} x r2 as r1 x n_k x n_{k+1} r2, same memory layout, so that
      //  the left unfolding of the update splits off core k)
      if( nMALS == 2 )
        C = fold_right(unfold_right(std::move(C)), dims[k]);
      CoreUpdate<T> upd = updateCoreLeft<T>(C, leftInterfaceOfY(k), localTolerance, maxRanks[k], maxvolTolerance, std::move(E),
                                            oversampling, options.enrichmentRank, maxRanks[k], maxIndices[k]);
      I[k] = leftIndicesFromRows(upd.selected, B.r1());
      fx.invalidate(k+1);

      // new core(s): core k left-orthonormal; the carry keeps y consistent as a tensor until the next core is rebuilt
      std::vector<Tensor3<T>> newCores(2);
      newCores[0] = std::move(upd.core);
      if( nMALS == 2 )
        newCores[1] = fold_right(std::move(upd.carry), dims[k+1]);
      else
        internal::normalize_contract1(upd.carry, TTy.subTensor(k+1), newCores[1]);
      TTy.setSubTensors(k, std::move(newCores), {TT_Orthogonality::left, TT_Orthogonality::none});
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
        Iz[k] = leftIndicesFromRows(dominantRows(residual, localTolerance, options.errorRank, maxvolTolerance), residual.r1());
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
      const Tensor3<T> B = fx.blockFromRight(kLeft, nMALS);
      rmsRef = std::max(rmsRef, t3_rms(B));
      const RealType scale = rmsRef > 0 ? rmsRef : 1;

      Ly.update(kLeft-1, k+1);
      RealType change = rmsDifference(B, yOnOwnCross(kLeft)) / scale;
      RealType error = 0;

      Tensor3<T> C = toCoordinatesOfY(kLeft, B);

      // first core of the 1-site sweep: the block is the core
      if( nMALS == 1 && k == 0 )
      {
        TTy.setSubTensor(k, std::move(C));
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
        C = fold_left(unfold_left(std::move(C)), dims[k]);
      CoreUpdate<T> upd = updateCoreRight<T>(C, rightInterfaceOfY(kLeft), localTolerance, maxRanks[k-1], maxvolTolerance, oversampling, maxIndices[k-1]);
      J[k] = rightIndicesFromColumns(upd.selected, dims[k]);
      fx.invalidate(k-1);

      // new core(s): core k right-orthonormal, the carry multiplied into the left neighbor
      std::vector<Tensor3<T>> newCores(2);
      if( nMALS == 2 )
        newCores[0] = fold_left(std::move(upd.carry), dims[k-1]);
      else
        internal::normalize_contract2(TTy.subTensor(k-1), upd.carry, newCores[0]);
      newCores[1] = std::move(upd.core);
      TTy.setSubTensors(k-1, std::move(newCores), {TT_Orthogonality::none, TT_Orthogonality::right});
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
        Jz[k] = rightIndicesFromColumns(dominantColumns(residual, localTolerance, options.errorRank, maxvolTolerance), dims[k]);
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
