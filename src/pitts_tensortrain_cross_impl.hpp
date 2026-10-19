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
* The state of the algorithm and its local steps are internal::tt_cross::CrossSweepState; the sweep driver
* crossSweeps only decides the order of the steps and when to stop, so that other sweep orders can reuse the steps.
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
  //! State of the TT-cross approximation TTy ~ f(TTx[0], ..., TTx[N-1]) and the steps of its sweeps
  //!
  //! Everything of the TT-cross algorithm except the order of the steps: argument checks, initial frame and index
  //! sets, the cached partial contractions, one local step in each direction and the independent control sample.
  //! The sweep drivers (crossSweeps) only decide which steps are done in which order and when to stop.
  //!
  //! TTy and the index sets are referred to by the cached contractions, so the object can neither be copied nor moved.
  //!
  template<typename T, typename Function, std::size_t N>
  class CrossSweepState final
  {
    public:
      using RealType = decltype(std::abs(T()));
      using Inputs = std::array<std::reference_wrapper<const TensorTrain<T>>, N>;

      //! check the arguments and initialize y (a zero tensor with the right frame of the inputs) and the index sets
      CrossSweepState(const Function& f, const Inputs& TTx, TensorTrain<T>& TTy, const TTCrossOptions<T>& options) :
        options_(checkedOptions(TTx, TTy, options)),
        dims_(TTy.dimensions()),
        nDim_(dims_.size()),
        nMALS_(std::min(options.nMALS, nDim_)),
        // enrichment only for the 1-site variant (the 2-site variant adapts the ranks through the SVD)
        enrich_(nMALS_ == 1 && nDim_ > 1 && options.errorRank > 0 && options.enrichmentRank > 0),
        localTolerance_(options.rankTolerance / (2*std::sqrt(T(std::max(nDim_-1, 1))))),     // factor 2 heuristic, like tt_amen in ttpy
        // maximal useful rank per bond (also after the enrichment)
        maxRanks_(limitByNeighbors(options.maxRanks.empty() ?
          std::vector<long long>(std::max(nDim_-1, 0), options.maxRank) : std::vector<long long>(options.maxRanks.begin(), options.maxRanks.end()))),
        // maximal size of the index sets per bond: the oversampled maximal rank, again limited by the neighbors
        maxIndices_(limitByNeighbors(oversampled(maxRanks_, options.oversampling))),
        TTy_(TTy),
        I_(nDim_),
        // y is kept in the orthogonal gauge, like the solution in solveMALS: in a sweep, the cores left of the current
        // block are left-orthonormal and the cores right of it right-orthonormal, so the truncated SVD of a local block
        // is the truncation of y itself. Start: y = 0 with an orthonormal right frame spanned by the inputs, and
        // initial right index sets from a maxvol sweep over this frame.
        J_(initialRightIndexSets(TTx, TTy)),
        // partial contractions of y on its own crosses: their row/column selections are the interface matrices of y
        // (the values of the orthonormal left/right parts on I_k resp. J_k), which convert f(x) on a cross into the
        // coordinates of y (least squares with oversampling) and y back onto the cross
        Ly_(defineSweepData<Tensor3<T>>(nDim_, contraction_loop_from_left<T>(TTy, I_), contraction_loop_from_right<T>(TTy, J_))),
        // index sets of the error approximation z (no cores needed, only the crosses)
        Iz_(nDim_),
        Jz_(initialErrorIndexSets(TTx)),
        // f(x) on the crosses of y and z (with the partial contractions of x on them), partial contractions of y on the crosses of z
        fx_(f, TTx, I_, J_, Iz_, Jz_),
        Lyz_(defineSweepData<Tensor3<T>>(nDim_, contraction_loop_from_left<T>(TTy, Iz_), contraction_loop_from_right<T>(TTy, Jz_))),
        // independent control sample: random entries, drawn once so that the estimate is comparable between
        // sweeps; it does not enter the convergence check, see TTCrossOptions::controlSampleSize
        sampleIndices_(options.controlSampleSize > 0 ? randomIndices(dims_, options.controlSampleSize) : Eigen::MatrixXi(0, nDim_)),
        // f(x) on the sample does not change (fixed entries), so it is evaluated only once
        fOnSample_(sampleIndices_.rows() > 0 ? fx_.atIndices(sampleIndices_) : Tensor3<T>()),
        rmsfOnSample_(sampleIndices_.rows() > 0 ? t3_rms(fOnSample_) : RealType(0))
      {
      }

      CrossSweepState(const CrossSweepState&) = delete;
      CrossSweepState& operator=(const CrossSweepState&) = delete;

      //! number of dimensions
      [[nodiscard]] int nDim() const {return nDim_;}

      //! number of cores per local step (options.nMALS, at most nDim)
      [[nodiscard]] int nMALS() const {return nMALS_;}

      //! total number of evaluated tensor entries
      [[nodiscard]] long long nEvaluations() const {return fx_.nEvaluations();}

      //! one step of a left-to-right sweep at the cores swpIdx.leftDim()..swpIdx.rightDim()
      //!
      //! @return the convergence measures (change, error) of the step, see TTCrossInfo
      //!
      std::pair<RealType, RealType> stepLeftToRight(const SweepIndex& swpIdx)
      {
        const int k = swpIdx.leftDim();
        const int nMALS = nMALS_;
        auto& TTy = TTy_;

        // block B = f( x(I_{k-1}, S_k [x S_{k+1}], J_{kRight+1}) ) = f( LX_k [X_{k+1}] R_{kRight+1} );
        // LX_k = L_{k-1} X_k is cached: it is reused for the enrichment residual and for the next interface L_k
        const Tensor3<T> B = fx_.blockFromLeft(k, nMALS);
        rmsRef_ = std::max(rmsRef_, t3_rms(B));
        const RealType scale = rmsRef_ > 0 ? rmsRef_ : 1;   // zero blocks only: the measures become absolute

        // convergence measure: the current y on the same cross
        Ly_.update(k-1, k+nMALS);
        const auto [L, R] = interfacesOfY(k);
        RealType change = rmsDifference(B, onCross(L, combineCores(TTy, k, nMALS), R)) / scale;
        RealType error = 0;

        Tensor3<T> C = toCoordinates(L, B, R);

        // last core of the 1-site sweep: the block is the core, nothing to select
        if( nMALS == 1 && k == nDim_-1 )
        {
          TTy.setSubTensor(k, std::move(C));
          Lyz_.invalidate(k);   // the core of y changed
          Ly_.invalidate(k);
          return {change, error};
        }

        // enrichment: dominant directions of the error f(x) - y on the cross (I_{k-1}, S_k, J^z_{k+1}),
        // i.e. of f(LX_k R^z_{k+1}) - LY_k R^{yz}_{k+1}, in the left coordinates of y
        Tensor3<T> E;   // empty unless this step enriches
        if( enrich_ )
        {
          Tensor3<T> residual = fx_.mixedBlock(k);
          Lyz_.update(k-1, k+1);
          Tensor3<T> withLeft, yOnMixedCross;
          internal::normalize_contract1(L, TTy.subTensor(k), withLeft);
          contractRightInterface<T>(withLeft, Lyz_.right(), Jz_, k+1, yOnMixedCross);
          internal::t3_axpy(T(-1), yOnMixedCross, residual);
          error = std::max(error, t3_rms(residual) / scale);
          // as in solveMALS, updateCoreLeft first projects out the current directions and then appends the
          // dominant (at most enrichmentRank) new directions of the error
          // (not below the truncation tolerance: those directions would be truncated anyway, near convergence they are round-off)
          if( t3_rms(residual) / scale > std::real(localTolerance_) )
            solveLeftInterface(L, residual, E);
        }

        // local update: truncated SVD, enrichment, maxvol
        // (2-site: view the block r1 x n_k n_{k+1} x r2 as r1 x n_k x n_{k+1} r2, same memory layout, so that
        //  the left unfolding of the update splits off core k)
        if( nMALS == 2 )
          C = fold_right(unfold_right(std::move(C)), dims_[k]);
        CoreUpdate<T> upd = updateCoreLeft<T>(C, L, localTolerance_, maxRanks_[k], options_.maxvolTolerance, std::move(E),
                                              options_.oversampling, options_.enrichmentRank, maxRanks_[k], maxIndices_[k]);
        I_[k] = leftIndicesFromRows(upd.selected, B.r1());
        fx_.invalidate(k+1);

        // new core(s): core k left-orthonormal; the carry keeps y consistent as a tensor until the next core is rebuilt
        std::vector<Tensor3<T>> newCores(2);
        newCores[0] = std::move(upd.core);
        if( nMALS == 2 )
          newCores[1] = fold_right(std::move(upd.carry), dims_[k+1]);
        else
          internal::normalize_contract1(upd.carry, TTy.subTensor(k+1), newCores[1]);
        TTy.setSubTensors(k, std::move(newCores), {TT_Orthogonality::left, TT_Orthogonality::none});
        Ly_.invalidate(k, 2);

        // update of the error crosses: I^z_k from the error f(x) - y on (I^z_{k-1}, S_k, J^z_{k+1}),
        // i.e. of f(LX^z_k R^z_{k+1}) - LY^z_k R^{yz}_{k+1}
        if( enrich_ )
        {
          Lyz_.invalidate(k, 2);
          Tensor3<T> residual = fx_.errorBlockFromLeft(k);
          Lyz_.update(k, k+1);
          Tensor3<T> yOnCross;
          contractRightInterface<T>(Lyz_.left()->get(), Lyz_.right(), Jz_, k+1, yOnCross);
          internal::t3_axpy(T(-1), yOnCross, residual);
          error = std::max(error, t3_rms(residual) / scale);
          if( !keepErrorCross(Iz_[k], t3_rms(residual) / scale, residual.r1()) )
            Iz_[k] = leftIndicesFromRows(dominantRows(residual, errorTolerance(residual, scale), options_.errorRank, options_.maxvolTolerance), residual.r1());
          fx_.invalidateError(k+1);
          Lyz_.invalidate(k+1);
        }

        return {change, error};
      }

      //! one step of a right-to-left sweep at the cores swpIdx.leftDim()..swpIdx.rightDim() (mirror image of stepLeftToRight)
      std::pair<RealType, RealType> stepRightToLeft(const SweepIndex& swpIdx)
      {
        const int k = swpIdx.rightDim();
        const int kLeft = swpIdx.leftDim();
        const int nMALS = nMALS_;
        auto& TTy = TTy_;

        // block B = f( x(I_{kLeft-1}, [S_{k-1} x] S_k, J_{k+1}) ) = f( L_{kLeft-1} [X_{k-1}] XR_k ) with the cached XR_k = X_k R_{k+1}
        const Tensor3<T> B = fx_.blockFromRight(kLeft, nMALS);
        rmsRef_ = std::max(rmsRef_, t3_rms(B));
        const RealType scale = rmsRef_ > 0 ? rmsRef_ : 1;

        Ly_.update(kLeft-1, k+1);
        const auto [L, R] = interfacesOfY(kLeft);
        RealType change = rmsDifference(B, onCross(L, combineCores(TTy, kLeft, nMALS), R)) / scale;
        RealType error = 0;

        Tensor3<T> C = toCoordinates(L, B, R);

        // first core of the 1-site sweep: the block is the core
        if( nMALS == 1 && k == 0 )
        {
          TTy.setSubTensor(k, std::move(C));
          Lyz_.invalidate(k);
          Ly_.invalidate(k);
          return {change, error};
        }

        // No enrichment in this direction: the right-to-left sweep only truncates, so that after each full
        // sweep the ranks of y are the truncated ranks (not rank + enrichmentRank as in amen_cross, which
        // leaves the final truncation to the caller). Rank growth still happens: the enlarged left sets
        // I_{k-1} of the preceding left-to-right sweep provide the additional rows of the block here.
        // (2-site: view the block r1 x n_{k-1} n_k x r2 as r1 n_{k-1} x n_k x r2 for splitting off core k)
        if( nMALS == 2 )
          C = fold_left(unfold_left(std::move(C)), dims_[k]);
        CoreUpdate<T> upd = updateCoreRight<T>(C, R, localTolerance_, maxRanks_[k-1], options_.maxvolTolerance, options_.oversampling, maxIndices_[k-1]);
        J_[k] = rightIndicesFromColumns(upd.selected, dims_[k]);
        fx_.invalidate(k-1);

        // new core(s): core k right-orthonormal, the carry multiplied into the left neighbor
        std::vector<Tensor3<T>> newCores(2);
        if( nMALS == 2 )
          newCores[0] = fold_left(std::move(upd.carry), dims_[k-1]);
        else
          internal::normalize_contract2(TTy.subTensor(k-1), upd.carry, newCores[0]);
        newCores[1] = std::move(upd.core);
        TTy.setSubTensors(k-1, std::move(newCores), {TT_Orthogonality::none, TT_Orthogonality::right});
        Ly_.invalidate(k-1, 2);

        // update of the error crosses: J^z_k from the error f(x) - y on (I^z_{k-1}, S_k, J^z_{k+1}),
        // i.e. of f(L^z_{k-1} XR^z_k) - L^{yz}_{k-1} YR^z_k
        if( enrich_ )
        {
          Lyz_.invalidate(k-1, 2);
          Tensor3<T> residual = fx_.errorBlockFromRight(k);
          Lyz_.update(k-1, k);
          Tensor3<T> yOnCross;
          contractLeftInterface<T>(Lyz_.left(), Iz_, k-1, Lyz_.right()->get(), yOnCross);
          internal::t3_axpy(T(-1), yOnCross, residual);
          error = std::max(error, t3_rms(residual) / scale);
          if( !keepErrorCross(Jz_[k], t3_rms(residual) / scale, residual.r2()) )
            Jz_[k] = rightIndicesFromColumns(dominantColumns(residual, errorTolerance(residual, scale), options_.errorRank, options_.maxvolTolerance), dims_[k]);
          fx_.invalidateError(k-1);
          Lyz_.invalidate(k-1);
        }

        return {change, error};
      }

      //! relative error of y on the control sample (0 without control sample), see TTCrossInfo::lastSampleError
      [[nodiscard]] RealType sampleError() const
      {
        if( sampleIndices_.rows() == 0 )
          return 0;
        Tensor3<T> yOnSample;
        evaluateAt(TTy_, sampleIndices_, yOnSample);
        return rmsDifference(fOnSample_, yOnSample) / (rmsfOnSample_ > 0 ? rmsfOnSample_ : 1);
      }

      //! store the number of evaluations and the final index sets as full multi-indices in info
      void finalInfo(TTCrossInfo<T>& info) const
      {
        info.nEvaluations = nEvaluations();
        info.leftIndices.resize(std::max(nDim_-1, 0));
        for(int k = 0; k < nDim_-1; k++)
          info.leftIndices[k] = expandLeft(I_, k);
        info.rightIndices.resize(nDim_);
        for(int k = 1; k < nDim_; k++)
          info.rightIndices[k] = expandRight(J_, k, nDim_);
      }

    private:
      //! the options after checking them together with the dimensions of the tensors
      static const TTCrossOptions<T>& checkedOptions(const Inputs& TTx, const TensorTrain<T>& TTy, const TTCrossOptions<T>& options)
      {
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
        return options;
      }

      //! Limits per bond k (between cores k and k+1): a TT rank at bond k is at most n_k times the rank at bond k-1 and
      //! n_{k+1} times the rank at bond k+1, so a small bond also limits its neighbors (one pass in each direction suffices).
      std::vector<int> limitByNeighbors(std::vector<long long> limit) const
      {
        for(int k = 0; k < nDim_-1; k++)
          limit[k] = std::min(limit[k], dims_[k] * (k == 0 ? 1 : limit[k-1]));
        for(int k = nDim_-2; k >= 0; k--)
          limit[k] = std::min(limit[k], dims_[k+1] * (k == nDim_-2 ? 1 : limit[k+1]));
        return std::vector<int>(limit.begin(), limit.end());
      }

      //! the ranks enlarged by the oversampling factor
      static std::vector<long long> oversampled(const std::vector<int>& ranks, T oversampling)
      {
        std::vector<long long> result(ranks.begin(), ranks.end());
        for(long long& q: result)
          q += static_cast<long long>(std::ceil(std::real(oversampling) * q));
        return result;
      }

      //! initialize TTy with the right frame of the inputs and return the right index sets from this frame
      std::vector<NestedIndices> initialRightIndexSets(const Inputs& TTx, TensorTrain<T>& TTy) const
      {
        const std::vector<std::reference_wrapper<const TensorTrain<T>>> inputs(TTx.begin(), TTx.end());
        initialRightFrame(inputs, TTy);
        return rightIndexSetsFromFrame(TTy, options_.maxvolTolerance, options_.oversampling, maxIndices_);
      }

      //! initial right index sets of the error crosses (empty without enrichment)
      std::vector<NestedIndices> initialErrorIndexSets(const Inputs& TTx) const
      {
        if( !enrich_ )
          return std::vector<NestedIndices>(nDim_);
        if( options_.randomErrorIndexSets )
          return randomRightIndexSets(dims_, options_.errorRank);
        // maxvol sets of a frame of rank errorRank of the inputs: well-conditioned crosses of the dominant part of x
        const std::vector<std::reference_wrapper<const TensorTrain<T>>> inputs(TTx.begin(), TTx.end());
        TensorTrain<T> TTz(dims_);
        initialRightFrame(inputs, TTz, options_.errorRank);
        return rightIndexSetsFromFrame(TTz, options_.maxvolTolerance);
      }

      //! the 1x1 identity: interface at the boundary
      static Tensor2<T> identity1()
      {
        Tensor2<T> one(1, 1);
        one(0, 0) = T(1);
        return one;
      }

      //! The interface matrices of y at the cross (I_{k-1}, S_k x ... x S_{k+nMALS-1}, J_{k+nMALS}) of the block of the
      //! cores k..k+nMALS-1 (the 1x1 identity at the boundary; needs Ly.update(k-1, k+nMALS) first)
      std::pair<Tensor2<T>, Tensor2<T>> interfacesOfY(int k) const
      {
        return {Ly_.left() ? interfaceLeft(Ly_.left()->get(), I_[k-1]) : identity1(),
                Ly_.right() ? interfaceRight(Ly_.right()->get(), J_[k+nMALS_]) : identity1()};
      }

      //! core(s) of y onto the cross, L X R
      static Tensor3<T> onCross(const Tensor2<T>& L, const Tensor3<T>& X, const Tensor2<T>& R)
      {
        Tensor3<T> withLeft, result;
        internal::normalize_contract1(L, X, withLeft);
        internal::normalize_contract2(withLeft, R, result);
        return result;
      }

      //! a block on the cross in the coordinates of y, L^+ B R^+
      static Tensor3<T> toCoordinates(const Tensor2<T>& L, const Tensor3<T>& B, const Tensor2<T>& R)
      {
        Tensor3<T> withLeft, result;
        solveLeftInterface(L, B, withLeft);
        solveRightInterface(withLeft, R, result);
        return result;
      }

      // Error crosses: fixed size errorRank (as zrank in amen_cross), selected from the directions of the residual above
      // the local tolerance (in the Frobenius norm of the block, i.e. rms above localTolerance * scale). No decisions on
      // round-off: a numerical rank of the residual would change the size of the crosses between runs, and below the
      // tolerance the residual is round-off near convergence, so the error crosses are kept then. A kept set must only
      // refer to existing positions of its (possibly updated) parent set.
      T errorTolerance(const Tensor3<T>& residual, RealType scale) const
      {
        return T(std::real(localTolerance_) * scale * std::sqrt(RealType(residual.r1()*residual.n()*residual.r2())));
      }
      bool keepErrorCross(const NestedIndices& Z, RealType relativeError, int parentSize) const
      {
        return relativeError <= std::real(localTolerance_) && Z.size() > 0 && *std::max_element(Z.link.begin(), Z.link.end()) < parentSize;
      }

      const TTCrossOptions<T>& options_;
      const std::vector<int>& dims_;
      const int nDim_;
      const int nMALS_;
      const bool enrich_;
      const T localTolerance_;
      const std::vector<int> maxRanks_;
      const std::vector<int> maxIndices_;
      TensorTrain<T>& TTy_;
      //! index sets of y: I[k] for k = 0..nDim-2, J[k] for k = 1..nDim-1 (J[0] unused)
      std::vector<NestedIndices> I_;
      std::vector<NestedIndices> J_;
      //! partial contractions of y on its own crosses
      PartialContractions<T> Ly_;
      //! index sets of the error approximation z
      std::vector<NestedIndices> Iz_;
      std::vector<NestedIndices> Jz_;
      FunctionOnCrosses<T, Function, N> fx_;
      //! partial contractions of y on the crosses of z
      PartialContractions<T> Lyz_;
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
      RealType rmsRef_ = 0;
      const Eigen::MatrixXi sampleIndices_;
      const Tensor3<T> fOnSample_;
      const RealType rmsfOnSample_;
  };


  //! TT-cross sweeps for TTy ~ f(TTx[0], ..., TTx[N-1]) element-wise, see applyElementwise
  template<typename T, typename Function, std::size_t N>
  TTCrossInfo<T> crossSweeps(const Function& f, const std::array<std::reference_wrapper<const TensorTrain<T>>, N>& TTx,
                             TensorTrain<T>& TTy, const TTCrossOptions<T>& options)
  {
    using RealType = typename CrossSweepState<T, Function, N>::RealType;

    CrossSweepState<T, Function, N> state(f, TTx, TTy, options);
    TTCrossInfo<T> info;

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

    const SweepIndex firstIdx(state.nDim(), state.nMALS(), state.nMALS()-1, 0);
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
        const auto [change, error] = state.stepLeftToRight(swpIdx);
        maxChange = std::max(maxChange, change);
        maxError = std::max(maxError, error);
      }
      info.lastSampleError = state.sampleError();
      if( options.verbose )
        printProgress(iSweep+0.5, maxChange, maxError);

      // right to left (this also truncates the enrichment of the left-to-right sweep, so the convergence
      // check only makes sense on the ranks after the full sweep)
      maxChange = maxError = 0;
      for(SweepIndex swpIdx = firstIdx.last(); swpIdx; swpIdx = swpIdx.previous())
      {
        const auto [change, error] = state.stepRightToLeft(swpIdx);
        maxChange = std::max(maxChange, change);
        maxError = std::max(maxError, error);
      }
      info.nSweeps = iSweep+1;
      info.lastChange = maxChange;
      info.lastError = maxError;
      info.lastSampleError = state.sampleError();
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

    state.finalInfo(info);
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
