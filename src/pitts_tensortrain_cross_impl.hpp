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
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "pitts_tensortrain_cross.hpp"
#include "pitts_tensortrain_cross_helper.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_sweep_data.hpp"
#include "pitts_tensortrain_sweep_index.hpp"
#include "pitts_tensor3_combine.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_random.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! dedicated helper functionality for TT-cross algorithms
    namespace tt_cross
    {
      //! random right-nested index sets J[k] (k = 1..d-1) with at most `size` elements each, drawn without replacement
      inline std::vector<NestedIndices> randomRightIndexSets(const std::vector<int>& dims, int size)
      {
        const int nDim = dims.size();
        std::vector<NestedIndices> J(nDim);
        for(int k = nDim-1; k >= 1; k--)
        {
          const int childSize = (k == nDim-1) ? 1 : J[k+1].size();
          // all (idx, link) candidates in a shuffled order, take the first ones
          std::vector<int> candidates(dims[k]*childSize);
          std::iota(candidates.begin(), candidates.end(), 0);
          std::shuffle(candidates.begin(), candidates.end(), randomGenerator);
          candidates.resize(std::min<int>(size, candidates.size()));
          J[k] = rightIndicesFromColumns(candidates, dims[k]);
        }
        return J;
      }

      //! LX = L * X (all mode indices) or X itself at the left boundary
      template<typename T>
      void contractLeft(optional_cref<Tensor2<T>> L, const Tensor3<T>& X, Tensor3<T>& LX)
      {
        if( L )
          normalize_contract1(L->get(), X, LX);
        else
          copy(X, LX);
      }

      //! XR = X * R (all mode indices) or X itself at the right boundary
      template<typename T>
      void contractRight(const Tensor3<T>& X, optional_cref<Tensor2<T>> R, Tensor3<T>& XR)
      {
        if( R )
          normalize_contract2(X, R->get(), XR);
        else
          copy(X, XR);
      }

      //! Frobenius norm of a block
      template<typename T>
      auto blockNorm(const Tensor3<T>& B)
      {
        return ConstEigenMap(unfold_left(B)).norm();
      }

      //! B -= A for blocks of equal shape
      template<typename T>
      void subtractBlock(const Tensor3<T>& A, Tensor3<T>& B)
      {
        assert(A.r1() == B.r1() && A.n() == B.n() && A.r2() == B.r2());
        EigenMap(unfold_left(B)) -= ConstEigenMap(unfold_left(A));
      }

      //! view the 2-core block (r1 x n1*n2 x r2) as (r1 x n1 x n2*r2) (leftOrthog) or (r1*n1 x n2 x r2), same memory layout
      template<typename T>
      void reshapeSupercore(Tensor3<T>& B, int n1, int n2, bool leftOrthog)
      {
        assert(B.n() == n1*n2);
        if( leftOrthog )
          B.resize(B.r1(), n1, n2*B.r2(), false, true);
        else
          B.resize(B.r1()*n1, n2, B.r2(), false, true);
      }
    }
  }


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

    // index sets of y: I[k] for k = 0..nDim-2, J[k] for k = 1..nDim-1 (J[0] unused); interfaces of x on them
    std::vector<NestedIndices> I(nDim);
    std::vector<NestedIndices> J = initialRightIndexSets(TTx, maxvolTolerance);
    auto Lx = defineSweepData<Tensor2<T>>(nDim, interface_loop_from_left<T>(TTx, I), interface_loop_from_right<T>(TTx, J));

    // index sets of the error approximation z (no cores needed, only the crosses), interfaces of x and y on them
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
    auto Lxz = defineSweepData<Tensor2<T>>(nDim, interface_loop_from_left<T>(TTx, Iz), interface_loop_from_right<T>(TTx, Jz));
    auto Lyz = defineSweepData<Tensor2<T>>(nDim, interface_loop_from_left<T>(TTy, Iz), interface_loop_from_right<T>(TTy, Jz));

    // y is undefined until the first left-to-right sweep has built all cores; until then the residual is f(x) itself
    bool yValid = false;
    std::vector<Tensor3<T>> firstCores(nDim);

    // dominant `enrichmentRank` directions of a residual block (columns for leftOrthog, rows otherwise)
    const auto dominantDirections = [&](Tensor3<T>&& residual, bool leftOrthog) -> Tensor2<T>
    {
      if( leftOrthog )
      {
        auto [U, SVt] = internal::normalize_svd(unfold_left(residual), true, T(0), options.enrichmentRank);
        return std::move(U);
      }
      auto [US, Vt] = internal::normalize_svd(unfold_right(residual), false, T(0), options.enrichmentRank);
      return std::move(Vt);
    };

    // Convergence measure of a step: the relative change of y on its own cross (block vs. current core)
    // and, with enrichment, the relative error f(x) - y on the error crosses. The change alone is blind
    // when the crosses of y are too small but y is exact on them (e.g. a rank that is too low at some
    // bond); the residual on the (different) error crosses detects this and prevents an early stop.
    const auto relativeNorm = [](const Tensor3<T>& R, RealType normB) -> RealType
    {
      const RealType nrm = blockNorm(R);
      return normB > 0 ? nrm / normB : nrm;
    };

    // ---- one step of a left-to-right sweep at the cores swpIdx.leftDim()..swpIdx.rightDim() ----
    // returns the convergence measure of the step (0 in the first sweep)
    const auto stepLeftToRight = [&](const SweepIndex& swpIdx) -> RealType
    {
      const int k = swpIdx.leftDim();
      const int kRight = swpIdx.rightDim();
      Lx.update(k-1, kRight+1);

      // block B = f( x(I_{k-1}, S_k [x S_{k+1}], J_{kRight+1}) ); LX = L_{k-1} X_k is reused for the enrichment
      Tensor3<T> LX, B;
      contractLeft<T>(Lx.left(), TTx.subTensor(k), LX);
      if( nMALS == 2 )
      {
        const Tensor3<T> LXX = combine(LX, TTx.subTensor(k+1));
        contractRight<T>(LXX, Lx.right(), B);
      }
      else
        contractRight<T>(LX, Lx.right(), B);
      info.nEvaluations += applyToBlock(f, B);
      const RealType normB = blockNorm(B);

      // convergence measure: the current y on the same cross is exactly the current core(s)
      RealType change = 0;
      if( yValid )
      {
        if( nMALS == 2 )
          change = relativeDifference(B, combine(TTy.subTensor(k), TTy.subTensor(k+1)));
        else
          change = relativeDifference(B, TTy.subTensor(k));
      }

      // last core of the 1-site sweep: the block is the core, nothing to select
      if( nMALS == 1 && k == nDim-1 )
      {
        if( yValid )
          TTy.setSubTensor(k, std::move(B));
        else
          firstCores[k] = std::move(B);
        return change;
      }

      // enrichment: dominant directions of the residual f(x) - y on the cross (I_{k-1}, S_k, J^z_{k+1})
      Tensor2<T> E;
      optional_cref<Tensor2<T>> enrichment;
      if( enrich )
      {
        Lxz.update(k-1, k+1);
        Tensor3<T> residual;
        contractRight<T>(LX, Lxz.right(), residual);
        info.nEvaluations += applyToBlock(f, residual);
        if( yValid )
        {
          // y on the cross: the left interface of y at I_{k-1} is the identity (interpolation gauge), so only the right part remains
          Lyz.update(k-1, k+1);
          Tensor3<T> yOnCross;
          contractRight<T>(TTy.subTensor(k), Lyz.right(), yOnCross);
          subtractBlock(yOnCross, residual);
          change = std::max(change, relativeNorm(residual, normB));
        }
        E = dominantDirections(std::move(residual), true);
        enrichment = std::cref(E);
      }

      // local update: truncated SVD, enrichment, maxvol, interpolation gauge
      if( nMALS == 2 )
        reshapeSupercore(B, dims[k], dims[k+1], true);
      const int rPrev = B.r1();
      CoreUpdate<T> upd = updateCoreLeft<T>(B, localTolerance, maxRank, maxvolTolerance, enrichment);
      I[k] = leftIndicesFromRows(upd.selected, rPrev);
      Lx.invalidate(k);

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

      // update of the error crosses: I^z_k from the residual on (I^z_{k-1}, S_k, J^z_{k+1})
      if( enrich )
      {
        Lyz.invalidate(k, 2);
        Lxz.update(k-1, k+1);
        Tensor3<T> LXz, residual;
        contractLeft<T>(Lxz.left(), TTx.subTensor(k), LXz);
        contractRight<T>(LXz, Lxz.right(), residual);
        info.nEvaluations += applyToBlock(f, residual);
        if( yValid )
        {
          Lyz.update(k-1, k+1);
          Tensor3<T> LYz, yOnCross;
          contractLeft<T>(Lyz.left(), TTy.subTensor(k), LYz);
          contractRight<T>(LYz, Lyz.right(), yOnCross);
          subtractBlock(yOnCross, residual);
          change = std::max(change, relativeNorm(residual, normB));
        }
        const int rPrevZ = residual.r1();
        const CoreUpdate<T> updz = updateCoreLeft<T>(residual, localTolerance, options.errorRank, maxvolTolerance);
        Iz[k] = leftIndicesFromRows(updz.selected, rPrevZ);
        Lxz.invalidate(k);
        Lyz.invalidate(k);
      }

      return change;
    };

    // ---- one step of a right-to-left sweep at the cores swpIdx.leftDim()..swpIdx.rightDim() (mirror image) ----
    const auto stepRightToLeft = [&](const SweepIndex& swpIdx) -> RealType
    {
      const int k = swpIdx.rightDim();
      const int kLeft = swpIdx.leftDim();
      Lx.update(kLeft-1, k+1);

      // block B = f( x(I_{kLeft-1}, [S_{k-1} x] S_k, J_{k+1}) ); XR = X_k R_{k+1} is reused for the enrichment
      Tensor3<T> XR, B;
      contractRight<T>(TTx.subTensor(k), Lx.right(), XR);
      if( nMALS == 2 )
      {
        const Tensor3<T> XXR = combine(TTx.subTensor(k-1), XR);
        contractLeft<T>(Lx.left(), XXR, B);
      }
      else
        contractLeft<T>(Lx.left(), XR, B);
      info.nEvaluations += applyToBlock(f, B);
      const RealType normB = blockNorm(B);

      RealType change = 0;
      if( nMALS == 2 )
        change = relativeDifference(B, combine(TTy.subTensor(k-1), TTy.subTensor(k)));
      else
        change = relativeDifference(B, TTy.subTensor(k));

      // first core of the 1-site sweep: the block is the core
      if( nMALS == 1 && k == 0 )
      {
        TTy.setSubTensor(k, std::move(B));
        return change;
      }

      // No enrichment in this direction: the right-to-left sweep only truncates, so that after each full
      // sweep the ranks of y are the truncated ranks (not rank + enrichmentRank as in amen_cross, which
      // leaves the final truncation to the caller). Rank growth still happens: the enlarged left sets
      // I_{k-1} of the preceding left-to-right sweep provide the additional rows of the block here.
      if( nMALS == 2 )
        reshapeSupercore(B, dims[k-1], dims[k], false);
      CoreUpdate<T> upd = updateCoreRight<T>(B, localTolerance, maxRank, maxvolTolerance);
      J[k] = rightIndicesFromColumns(upd.selected, dims[k]);
      Lx.invalidate(k);

      std::vector<Tensor3<T>> newCores(2);
      if( nMALS == 2 )
        newCores[0] = fold_left(std::move(upd.carry), dims[k-1]);
      else
        internal::normalize_contract2(TTy.subTensor(k-1), upd.carry, newCores[0]);
      newCores[1] = std::move(upd.core);
      TTy.setSubTensors(k-1, std::move(newCores));

      // update of the error crosses: J^z_k from the residual on (I^z_{k-1}, S_k, J^z_{k+1})
      if( enrich )
      {
        Lyz.invalidate(k-1, 2);
        Lxz.update(k-1, k+1);
        Tensor3<T> XRz, residual;
        contractRight<T>(TTx.subTensor(k), Lxz.right(), XRz);
        contractLeft<T>(Lxz.left(), XRz, residual);
        info.nEvaluations += applyToBlock(f, residual);
        Lyz.update(k-1, k+1);
        Tensor3<T> YRz, yOnCross;
        contractRight<T>(TTy.subTensor(k), Lyz.right(), YRz);
        contractLeft<T>(Lyz.left(), YRz, yOnCross);
        subtractBlock(yOnCross, residual);
        change = std::max(change, relativeNorm(residual, normB));
        const CoreUpdate<T> updz = updateCoreRight<T>(residual, localTolerance, options.errorRank, maxvolTolerance);
        Jz[k] = rightIndicesFromColumns(updz.selected, dims[k]);
        Lxz.invalidate(k);
        Lyz.invalidate(k);
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

      // right to left (convergence is only checked after the full sweep: the ranks are enriched now)
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
