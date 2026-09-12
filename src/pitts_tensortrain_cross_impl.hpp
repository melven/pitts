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
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_CROSS_IMPL_HPP
#define PITTS_TENSORTRAIN_CROSS_IMPL_HPP

// includes
#include <algorithm>
#include <cmath>
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
#include "pitts_tensor3_combine.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement applyElementwise
  template<typename T, typename Function>
  TTCrossInfo<T> applyElementwise(const Function& f, const TensorTrain<T>& TTx, TensorTrain<T>& TTy, const TTCrossOptions<T>& options)
  {
    using namespace internal::tt_cross;
    using internal::optional_cref;
    using internal::defineSweepData;
    using internal::SweepIndex;
    using RealType = decltype(std::abs(T()));

    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // check arguments
    const std::vector<int>& dims = TTx.dimensions();
    const int nDim = dims.size();
    if( nDim < 1 )
      throw std::invalid_argument("applyElementwise: empty tensor train!");
    if( TTy.dimensions() != dims )
      throw std::invalid_argument("applyElementwise: dimensions of TTx and TTy differ!");
    if( TTx.subTensor(0).r1() != 1 || TTx.subTensor(nDim-1).r2() != 1 )
      throw std::invalid_argument("applyElementwise: boundary ranks of TTx must be 1!");
    if( options.nMALS < 1 || options.nMALS > 2 )
      throw std::invalid_argument("applyElementwise: nMALS must be 1 or 2!");
    if( options.errorRank < 0 || options.enrichmentRank < 0 || options.enrichmentRank > options.errorRank )
      throw std::invalid_argument("applyElementwise: need 0 <= enrichmentRank <= errorRank!");
    if( options.nSweeps < 1 || options.maxRank < 1 )
      throw std::invalid_argument("applyElementwise: nSweeps and maxRank must be positive!");

    const int nMALS = std::min(options.nMALS, nDim);
    // enrichment only for the 1-site variant (the 2-site variant adapts the ranks through the SVD)
    const bool enrich = nMALS == 1 && nDim > 1 && options.errorRank > 0 && options.enrichmentRank > 0;
    const T localTolerance = options.rankTolerance / std::sqrt(T(std::max(nDim-1, 1)));
    const T maxvolTolerance = options.maxvolTolerance;
    const int maxRank = options.maxRank;

    TTCrossInfo<T> info;

    // index sets of y: I[k] for k = 0..nDim-2, J[k] for k = 1..nDim-1 (J[0] unused); partial contractions of x on them
    std::vector<NestedIndices> I(nDim);
    std::vector<NestedIndices> J = initialRightIndexSets(TTx, maxvolTolerance);
    auto Lx = defineSweepData<Tensor3<T>>(nDim, contraction_loop_from_left<T>(TTx, I), contraction_loop_from_right<T>(TTx, J));

    // index sets of the error approximation z (no cores needed, only the crosses), partial contractions of x and y on them
    std::vector<NestedIndices> Iz(nDim), Jz(nDim);
    if( enrich )
    {
      if( options.randomErrorIndexSets )
        Jz = randomRightIndexSets(dims, options.errorRank);
      else
      {
        // maxvol sets of a rank-errorRank truncation of x: well-conditioned crosses of the dominant part of x
        TensorTrain<T> TTxTrunc(dims);
        copy(TTx, TTxTrunc);
        normalize(TTxTrunc, T(0), options.errorRank);
        Jz = initialRightIndexSets(TTxTrunc, maxvolTolerance);
      }
    }
    auto Lxz = defineSweepData<Tensor3<T>>(nDim, contraction_loop_from_left<T>(TTx, Iz), contraction_loop_from_right<T>(TTx, Jz));
    auto Lyz = defineSweepData<Tensor3<T>>(nDim, contraction_loop_from_left<T>(TTy, Iz), contraction_loop_from_right<T>(TTy, Jz));

    // y is undefined until the first left-to-right sweep has built all cores; until then the residual is f(x) itself
    bool yValid = false;
    std::vector<Tensor3<T>> firstCores(nDim);

    // Convergence measure of a step: the relative change of y on its own cross (block vs. current core)
    // and, with enrichment, the relative error f(x) - y on the error crosses, both scaled by the norm of
    // the block. The change alone is blind when the crosses of y are too small but y is exact on them
    // (e.g. a rank that is too low at some bond); the error on the (different) error crosses detects this
    // and prevents an early stop.

    // ---- one step of a left-to-right sweep at the cores swpIdx.leftDim()..swpIdx.rightDim() ----
    // returns the convergence measure of the step (0 in the first sweep)
    const auto stepLeftToRight = [&](const SweepIndex& swpIdx) -> RealType
    {
      const int k = swpIdx.leftDim();
      const int kRight = swpIdx.rightDim();

      // block B = f( x(I_{k-1}, S_k [x S_{k+1}], J_{kRight+1}) ) = LX_k [X_{k+1}] R_{kRight+1};
      // LX_k = L_{k-1} X_k is cached: it is reused for the enrichment residual and for the next interface L_k
      Lx.update(k, kRight+1);
      const Tensor3<T>& LX = Lx.left()->get();
      Tensor3<T> B;
      localBlockFromLeft<T>(LX, TTx, k, nMALS, Lx.right(), J, B);
      info.nEvaluations += applyToBlock(f, B);
      RealType normB = internal::t3_nrm(B);
      if( normB == 0 )
        normB = 1;   // zero block: the measures below become absolute

      // convergence measure: the current y on the same cross is exactly the current core(s)
      RealType change = 0;
      if( yValid )
        change = relativeDifference(B, combineCores(TTy, k, nMALS));

      // last core of the 1-site sweep: the block is the core, nothing to select
      if( nMALS == 1 && k == nDim-1 )
      {
        if( yValid )
          TTy.setSubTensor(k, std::move(B));
        else
          firstCores[k] = std::move(B);
        Lyz.invalidate(k);   // the core of y changed
        return change;
      }

      // enrichment: dominant directions of the error f(x) - y on the cross (I_{k-1}, S_k, J^z_{k+1}),
      // i.e. of f(LX_k R^z_{k+1}) - Y_k R^{yz}_{k+1}
      Tensor2<T> E;
      optional_cref<Tensor2<T>> enrichment;
      if( enrich )
      {
        Lxz.update(k, k+1);
        Tensor3<T> residual;
        contractRightInterface<T>(LX, Lxz.right(), Jz, k+1, residual);
        info.nEvaluations += applyToBlock(f, residual);
        if( yValid )
        {
          // y on the cross: the left interface of y at I_{k-1} is the identity (interpolation gauge), so only the right part remains
          Lyz.update(k-1, k+1);
          Tensor3<T> yOnCross;
          contractRightInterface<T>(TTy.subTensor(k), Lyz.right(), Jz, k+1, yOnCross);
          internal::t3_axpy(T(-1), yOnCross, residual);
          change = std::max(change, internal::t3_nrm(residual) / normB);
        }
        // dominant directions of the error: its leading (at most enrichmentRank) left singular vectors
        E = internal::normalize_svd(unfold_left(std::move(residual)), true, T(0), options.enrichmentRank).first;
        enrichment = std::cref(E);
      }

      // local update: truncated SVD, enrichment, maxvol, interpolation gauge
      // (2-site: view the block r1 x n_k n_{k+1} x r2 as r1 x n_k x n_{k+1} r2, same memory layout, so that
      //  the left unfolding of the update splits off core k)
      if( nMALS == 2 )
        B = fold_right(unfold_right(std::move(B)), dims[k]);
      const int rPrev = B.r1();
      CoreUpdate<T> upd = updateCoreLeft<T>(B, localTolerance, maxRank, maxvolTolerance, enrichment);
      I[k] = leftIndicesFromRows(upd.selected, rPrev);
      Lx.invalidate(k+1);

      // new core(s); the carry keeps y consistent as a tensor until the next core is rebuilt
      if( nMALS == 2 )
      {
        Tensor3<T> next = fold_right(std::move(upd.carry), dims[k+1]);
        if( yValid )
        {
          std::vector<Tensor3<T>> newCores(2);
          newCores[0] = std::move(upd.core);
          newCores[1] = std::move(next);
          TTy.setSubTensors(k, std::move(newCores));
        }
        else
        {
          firstCores[k] = std::move(upd.core);
          firstCores[k+1] = std::move(next);
        }
      }
      else if( yValid )
      {
        std::vector<Tensor3<T>> newCores(2);
        newCores[0] = std::move(upd.core);
        internal::normalize_contract1(upd.carry, TTy.subTensor(k+1), newCores[1]);
        TTy.setSubTensors(k, std::move(newCores));
      }
      else
        firstCores[k] = std::move(upd.core);

      // update of the error crosses: I^z_k from the error f(x) - y on (I^z_{k-1}, S_k, J^z_{k+1}),
      // i.e. of f(LX^z_k R^z_{k+1}) - LY^z_k R^{yz}_{k+1}
      if( enrich )
      {
        Lyz.invalidate(k, 2);
        Tensor3<T> residual;
        contractRightInterface<T>(Lxz.left()->get(), Lxz.right(), Jz, k+1, residual);
        info.nEvaluations += applyToBlock(f, residual);
        if( yValid )
        {
          Lyz.update(k, k+1);
          Tensor3<T> yOnCross;
          contractRightInterface<T>(Lyz.left()->get(), Lyz.right(), Jz, k+1, yOnCross);
          internal::t3_axpy(T(-1), yOnCross, residual);
          change = std::max(change, internal::t3_nrm(residual) / normB);
        }
        const int rPrevZ = residual.r1();
        const CoreUpdate<T> updz = updateCoreLeft<T>(residual, localTolerance, options.errorRank, maxvolTolerance);
        Iz[k] = leftIndicesFromRows(updz.selected, rPrevZ);
        Lxz.invalidate(k+1);
        Lyz.invalidate(k+1);
      }

      return change;
    };

    // ---- one step of a right-to-left sweep at the cores swpIdx.leftDim()..swpIdx.rightDim() (mirror image) ----
    const auto stepRightToLeft = [&](const SweepIndex& swpIdx) -> RealType
    {
      const int k = swpIdx.rightDim();
      const int kLeft = swpIdx.leftDim();

      // block B = f( x(I_{kLeft-1}, [S_{k-1} x] S_k, J_{k+1}) ) = L_{kLeft-1} [X_{k-1}] XR_k with the cached XR_k = X_k R_{k+1}
      Lx.update(kLeft-1, k);
      const Tensor3<T>& XR = Lx.right()->get();
      Tensor3<T> B;
      localBlockFromRight<T>(Lx.left(), I, TTx, kLeft, nMALS, XR, B);
      info.nEvaluations += applyToBlock(f, B);
      RealType normB = internal::t3_nrm(B);
      if( normB == 0 )
        normB = 1;   // zero block: the measures below become absolute

      RealType change = relativeDifference(B, combineCores(TTy, kLeft, nMALS));

      // first core of the 1-site sweep: the block is the core
      if( nMALS == 1 && k == 0 )
      {
        TTy.setSubTensor(k, std::move(B));
        Lyz.invalidate(k);
        return change;
      }

      // No enrichment in this direction: the right-to-left sweep only truncates, so that after each full
      // sweep the ranks of y are the truncated ranks (not rank + enrichmentRank as in amen_cross, which
      // leaves the final truncation to the caller). Rank growth still happens: the enlarged left sets
      // I_{k-1} of the preceding left-to-right sweep provide the additional rows of the block here.
      // (2-site: view the block r1 x n_{k-1} n_k x r2 as r1 n_{k-1} x n_k x r2 for splitting off core k)
      if( nMALS == 2 )
        B = fold_left(unfold_left(std::move(B)), dims[k]);
      CoreUpdate<T> upd = updateCoreRight<T>(B, localTolerance, maxRank, maxvolTolerance);
      J[k] = rightIndicesFromColumns(upd.selected, dims[k]);
      Lx.invalidate(k-1);

      std::vector<Tensor3<T>> newCores(2);
      if( nMALS == 2 )
        newCores[0] = fold_left(std::move(upd.carry), dims[k-1]);
      else
        internal::normalize_contract2(TTy.subTensor(k-1), upd.carry, newCores[0]);
      newCores[1] = std::move(upd.core);
      TTy.setSubTensors(k-1, std::move(newCores));

      // update of the error crosses: J^z_k from the error f(x) - y on (I^z_{k-1}, S_k, J^z_{k+1}),
      // i.e. of f(L^z_{k-1} XR^z_k) - L^{yz}_{k-1} YR^z_k
      if( enrich )
      {
        Lyz.invalidate(k-1, 2);
        Lxz.update(k-1, k);
        Tensor3<T> residual;
        contractLeftInterface<T>(Lxz.left(), Iz, k-1, Lxz.right()->get(), residual);
        info.nEvaluations += applyToBlock(f, residual);
        Lyz.update(k-1, k);
        Tensor3<T> yOnCross;
        contractLeftInterface<T>(Lyz.left(), Iz, k-1, Lyz.right()->get(), yOnCross);
        internal::t3_axpy(T(-1), yOnCross, residual);
        change = std::max(change, internal::t3_nrm(residual) / normB);
        const CoreUpdate<T> updz = updateCoreRight<T>(residual, localTolerance, options.errorRank, maxvolTolerance);
        Jz[k] = rightIndicesFromColumns(updz.selected, dims[k]);
        Lxz.invalidate(k-1);
        Lyz.invalidate(k-1);
      }

      return change;
    };

    // ---- sweeps ----
    const auto printProgress = [&](double sweep, T maxChange)
    {
      std::cout << "TT-cross sweep " << sweep << ": max. relative change " << maxChange << ", ranks:";
      for(int r: TTy.getTTranks())
        std::cout << " " << r;
      std::cout << "\n";
    };

    const SweepIndex firstIdx(nDim, nMALS, nMALS-1, 0);
    for(int iSweep = 0; iSweep < options.nSweeps; iSweep++)
    {
      // left to right (the very first one builds y)
      RealType maxChange = 0;
      for(SweepIndex swpIdx = firstIdx.first(); swpIdx; swpIdx = swpIdx.next())
        maxChange = std::max(maxChange, stepLeftToRight(swpIdx));
      if( !yValid )
      {
        TTy.setSubTensors(0, std::move(firstCores));
        yValid = true;
        Lyz.invalidate(0, nDim);
      }
      else if( options.verbose )
        printProgress(iSweep+0.5, maxChange);

      // right to left (this also truncates the enrichment of the left-to-right sweep, so the convergence
      // check only makes sense on the ranks after the full sweep)
      maxChange = 0;
      for(SweepIndex swpIdx = firstIdx.last(); swpIdx; swpIdx = swpIdx.previous())
        maxChange = std::max(maxChange, stepRightToLeft(swpIdx));
      info.nSweeps = iSweep+1;
      info.lastChange = maxChange;
      if( options.verbose )
        printProgress(iSweep+1, maxChange);
      if( maxChange < options.convergenceTolerance )
      {
        info.converged = true;
        break;
      }
    }

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


#endif // PITTS_TENSORTRAIN_CROSS_IMPL_HPP
