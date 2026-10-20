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
#include <optional>
#include <iostream>
#include <stdexcept>
#include <type_traits>
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
  //! Symmetric mode (applyElementwiseSymmetric, see the symmetric variant in pitts_tensortrain_cross.hpp): TTy is the working
  //! half in the right-orthonormal form, the symmetric approximation G diag(lambda) G^T is held in sym_, and the right sets at
  //! and beyond the middle bond are the mirrored left sets.
  //!
  template<typename T, typename Function, std::size_t N>
  class CrossSweepState final
  {
    public:
      using RealType = decltype(std::abs(T()));
      using Inputs = std::array<std::reference_wrapper<const TensorTrain<T>>, N>;

      //! convergence measures of a local step (see TTCrossInfo): change of y on its own cross, error on the error crosses
      struct StepMeasures
      {
        RealType change = 0;
        RealType error = 0;
      };

      //! check the arguments and initialize y (a zero tensor with the right frame of the inputs) and the index sets
      //!
      //! All setup happens in the member initializers, in the order in which the members are declared: each initializer
      //! only uses members declared before it (the order of the declarations below matters). Four initializers have
      //! side effects: options_ checks the arguments (first, before anything else), J_ initializes TTy (and in symmetric
      //! mode sym_ and I_), Jz_ in symmetric mode also Iz_, and fOnSample_ evaluates f on the control sample.
      //!
      //! @param symmetric  symmetric mode: TTx are halves of symmetric tensors and TTy a half, see above
      //!
      CrossSweepState(const Function& f, const Inputs& TTx, TensorTrain<T>& TTy, const TTCrossOptions<T>& options, bool symmetric = false) :
        options_(checkedOptions(TTx, TTy, options, symmetric)),
        nCores_(TTy.dimensions().size()),
        dims_(symmetric ? mirroredDimensions(TTy.dimensions()) : TTy.dimensions()),
        nDim_(dims_.size()),
        nMALS_(std::min(options_.nMALS, nCores_)),
        // enrichment only for the 1-site variant (the 2-site variant adapts the ranks through the SVD)
        enrich_(nMALS_ == 1 && nCores_ > 1 && options_.errorRank > 0 && options_.enrichmentRank > 0),
        // factor 2 heuristic, like tt_amen in ttpy; the same tolerance per bond of the full tensor in both modes (symmetric: an
        // inner bond of the half is two bonds of y, see symmetricRound)
        localTolerance_(options_.rankTolerance / (2*std::sqrt(T(std::max(nDim_-1, 1))))),
        // maximal useful rank per bond (also after the enrichment)
        maxRanks_(limitByNeighbors(dims_, std::vector<long long>(std::max(nDim_-1, 0), options_.maxRank))),
        // maximal size of the index sets per bond: the oversampled maximal rank, again limited by the neighbors
        maxIndices_(limitByNeighbors(dims_, oversampled(maxRanks_, options_.oversampling))),
        inputs_(TTx.begin(), TTx.end()),
        TTy_(TTy),
        I_(nDim_),
        sym_(symmetric ? std::make_optional<SymmetricPart>(TTy.dimensions()) : std::nullopt),
        // y is kept in the orthogonal gauge, like the solution in solveMALS: in a sweep, the cores left of the current
        // block are left-orthonormal and the cores right of it right-orthonormal, so the truncated SVD of a local block
        // is the truncation of y itself. Start: y = 0 with an orthonormal right frame spanned by the inputs, and
        // initial right index sets from a maxvol sweep over this frame.
        J_(initializeYAndRightIndexSets()),
        // partial contractions of y on its own crosses: their row/column selections are the interface matrices of y
        // (the values of the orthonormal left/right parts on I_k resp. J_k), which convert f(x) on a cross into the
        // coordinates of y (least squares with oversampling) and y back onto the cross
        // (symmetric: TTy is the half, the right part ends with the mirrored G^T, the boundary matrix Ry)
        Ly_(defineSweepData<Tensor3<T>>(nCores_, contraction_loop_from_left<T>(TTy, I_), contraction_loop_from_right<T>(TTy, J_, yBoundary(false)))),
        // index sets of the error approximation z (no cores needed, only the crosses)
        Iz_(nDim_),
        Jz_(initialErrorIndexSets()),
        // f(x) on the crosses of y and z (with the partial contractions of x on them), partial contractions of y on the crosses of z
        fx_(f, TTx, I_, J_, Iz_, Jz_, symmetric),
        Lyz_(defineSweepData<Tensor3<T>>(nCores_, contraction_loop_from_left<T>(TTy, Iz_), contraction_loop_from_right<T>(TTy, Jz_, yBoundary(true)))),
        // independent control sample: random entries, drawn once so that the estimate is comparable between
        // sweeps; it does not enter the convergence check, see TTCrossOptions::controlSampleSize
        sampleIndices_(options_.controlSampleSize > 0 ? randomIndices(dims_, options_.controlSampleSize) : Eigen::MatrixXi(0, nDim_)),
        // f(x) on the sample does not change (fixed entries), so it is evaluated only once
        fOnSample_(sampleIndices_.rows() > 0 ? fx_.atIndices(sampleIndices_) : Tensor3<T>()),
        rmsfOnSample_(sampleIndices_.rows() > 0 ? t3_rms(fOnSample_) : RealType(0))
      {
        // symmetric: the boundary matrices of x and of y on the error crosses (those of y on its own sets are set with J_)
        if( sym_ )
          updateBoundaries();
      }

      CrossSweepState(const CrossSweepState&) = delete;
      CrossSweepState& operator=(const CrossSweepState&) = delete;

      //! number of dimensions
      [[nodiscard]] int nDim() const {return nDim_;}

      //! number of cores of TTy (nDim, or m in symmetric mode)
      [[nodiscard]] int nCores() const {return nCores_;}

      //! number of cores per local step (options.nMALS, at most the number of cores of TTy)
      [[nodiscard]] int nMALS() const {return nMALS_;}

      //! total number of evaluated tensor entries
      [[nodiscard]] long long nEvaluations() const {return fx_.nEvaluations();}

      //! one step of a left-to-right sweep at the cores swpIdx.leftDim()..swpIdx.rightDim()
      //!
      //! @return the convergence measures of the step
      //!
      StepMeasures stepLeftToRight(const SweepIndex& swpIdx)
      {
        const int k = swpIdx.leftDim();

        // block B = f( x(I_{k-1}, S_k [x S_{k+1}], J_{kRight+1}) ) = f( LX_k [X_{k+1}] R_{kRight+1} );
        // LX_k = L_{k-1} X_k is cached: it is reused for the enrichment residual and for the next interface L_k
        const Tensor3<T> B = fx_.blockFromLeft(k, nMALS_);
        rmsRef_ = std::max(rmsRef_, t3_rms(B));
        const RealType scale = rmsRef_ > 0 ? rmsRef_ : 1;   // zero blocks only: the measures become absolute

        // convergence measure: the current y on the same cross
        Ly_.update(k-1, k+nMALS_);
        const auto [L, R] = interfacesOfY(k);
        Tensor3<T> yOnCross;
        contractInterfaces(L, combineCores(TTy_, k, nMALS_), R, yOnCross);
        RealType change = rmsDifference(B, yOnCross) / scale;
        RealType error = 0;

        Tensor3<T> C;
        solveInterfaces(L, B, R, C);

        // last core of the 1-site sweep: the block is the core, nothing to select
        if( nMALS_ == 1 && k == nDim_-1 )
        {
          TTy_.setSubTensor(k, std::move(C));
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
          internal::normalize_contract1(L, TTy_.subTensor(k), withLeft);
          contractRightInterface<T>(withLeft, Lyz_.right(), Jz_, k+1, yOnMixedCross, yBoundary(true));
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
        if( nMALS_ == 2 )
          C = fold_right(unfold_right(std::move(C)), dims_[k]);
        CoreUpdate<T> upd = updateCoreLeft<T>(C, L, localTolerance_, maxRanks_[k], options_.maxvolTolerance, std::move(E),
                                              options_.oversampling, options_.enrichmentRank, maxRanks_[k], maxIndices_[k]);
        I_[k] = leftIndicesFromRows(upd.selected, B.r1());
        fx_.invalidate(k+1);

        // new core(s): core k left-orthonormal; the carry keeps y consistent as a tensor until the next core is rebuilt
        std::vector<Tensor3<T>> newCores(2);
        newCores[0] = std::move(upd.core);
        if( nMALS_ == 2 )
          newCores[1] = fold_right(std::move(upd.carry), dims_[k+1]);
        else
          internal::normalize_contract1(upd.carry, TTy_.subTensor(k+1), newCores[1]);
        TTy_.setSubTensors(k, std::move(newCores), {TT_Orthogonality::left, TT_Orthogonality::none});
        Ly_.invalidate(k, 2);

        // update of the error crosses (the cores k, k+1 of y changed)
        if( enrich_ )
        {
          Lyz_.invalidate(k, 2);
          error = std::max(error, updateErrorCrossLeft(k, scale));
        }

        return {change, error};
      }

      //! I^z_k from the error f(x) - y on (I^z_{k-1}, S_k, J^z_{k+1}), i.e. of f(LX^z_k R^z_{k+1}) - LY^z_k R^{yz}_{k+1}
      //!
      //! @return the error estimate on this cross (rms relative to scale)
      //!
      RealType updateErrorCrossLeft(int k, RealType scale)
      {
        Tensor3<T> residual = fx_.errorBlockFromLeft(k);
        Lyz_.update(k, k+1);
        Tensor3<T> yOnCross;
        contractRightInterface<T>(Lyz_.left()->get(), Lyz_.right(), Jz_, k+1, yOnCross, yBoundary(true));
        internal::t3_axpy(T(-1), yOnCross, residual);
        const RealType error = t3_rms(residual) / scale;
        if( !keepErrorCross(Iz_[k], error, residual.r1()) )
          Iz_[k] = leftIndicesFromRows(dominantRows(residual, errorTolerance(residual, scale), options_.errorRank, options_.maxvolTolerance), residual.r1());
        fx_.invalidateError(k+1);
        Lyz_.invalidate(k+1);
        return error;
      }

      //! one step of a right-to-left sweep at the cores swpIdx.leftDim()..swpIdx.rightDim() (mirror image of stepLeftToRight)
      StepMeasures stepRightToLeft(const SweepIndex& swpIdx)
      {
        const int k = swpIdx.rightDim();
        const int kLeft = swpIdx.leftDim();

        // block B = f( x(I_{kLeft-1}, [S_{k-1} x] S_k, J_{k+1}) ) = f( L_{kLeft-1} [X_{k-1}] XR_k ) with the cached XR_k = X_k R_{k+1}
        const Tensor3<T> B = fx_.blockFromRight(kLeft, nMALS_);
        rmsRef_ = std::max(rmsRef_, t3_rms(B));
        const RealType scale = rmsRef_ > 0 ? rmsRef_ : 1;

        Ly_.update(kLeft-1, k+1);
        const auto [L, R] = interfacesOfY(kLeft);
        Tensor3<T> yOnCross;
        contractInterfaces(L, combineCores(TTy_, kLeft, nMALS_), R, yOnCross);
        RealType change = rmsDifference(B, yOnCross) / scale;
        RealType error = 0;

        Tensor3<T> C;
        solveInterfaces(L, B, R, C);

        // first core of the 1-site sweep: the block is the core
        if( nMALS_ == 1 && k == 0 )
        {
          TTy_.setSubTensor(k, std::move(C));
          Lyz_.invalidate(k);
          Ly_.invalidate(k);
          return {change, error};
        }

        // No enrichment in this direction: the right-to-left sweep only truncates, so that after each full
        // sweep the ranks of y are the truncated ranks (not rank + enrichmentRank as in amen_cross, which
        // leaves the final truncation to the caller). Rank growth still happens: the enlarged left sets
        // I_{k-1} of the preceding left-to-right sweep provide the additional rows of the block here.
        // (2-site: view the block r1 x n_{k-1} n_k x r2 as r1 n_{k-1} x n_k x r2 for splitting off core k)
        if( nMALS_ == 2 )
          C = fold_left(unfold_left(std::move(C)), dims_[k]);
        CoreUpdate<T> upd = updateCoreRight<T>(C, R, localTolerance_, maxRanks_[k-1], options_.maxvolTolerance, options_.oversampling, maxIndices_[k-1]);
        J_[k] = rightIndicesFromColumns(upd.selected, dims_[k]);
        fx_.invalidate(k-1);

        // new core(s): core k right-orthonormal, the carry multiplied into the left neighbor
        std::vector<Tensor3<T>> newCores(2);
        if( nMALS_ == 2 )
          newCores[0] = fold_left(std::move(upd.carry), dims_[k-1]);
        else
          internal::normalize_contract2(TTy_.subTensor(k-1), upd.carry, newCores[0]);
        newCores[1] = std::move(upd.core);
        TTy_.setSubTensors(k-1, std::move(newCores), {TT_Orthogonality::none, TT_Orthogonality::right});
        Ly_.invalidate(k-1, 2);

        // update of the error crosses (the cores k-1, k of y changed)
        if( enrich_ )
        {
          Lyz_.invalidate(k-1, 2);
          error = std::max(error, updateErrorCrossRight(k, scale));
        }

        return {change, error};
      }

      //! J^z_k from the error f(x) - y on (I^z_{k-1}, S_k, J^z_{k+1}), i.e. of f(L^z_{k-1} XR^z_k) - L^{yz}_{k-1} YR^z_k
      //!
      //! @return the error estimate on this cross (rms relative to scale)
      //!
      RealType updateErrorCrossRight(int k, RealType scale)
      {
        Tensor3<T> residual = fx_.errorBlockFromRight(k);
        Lyz_.update(k-1, k);
        Tensor3<T> yOnCross;
        contractLeftInterface<T>(Lyz_.left(), Iz_, k-1, Lyz_.right()->get(), yOnCross);
        internal::t3_axpy(T(-1), yOnCross, residual);
        const RealType error = t3_rms(residual) / scale;
        if( !keepErrorCross(Jz_[k], error, residual.r2()) )
          Jz_[k] = rightIndicesFromColumns(dominantColumns(residual, errorTolerance(residual, scale), options_.errorRank, options_.maxvolTolerance), dims_[k]);
        fx_.invalidateError(k-1);
        Lyz_.invalidate(k-1);
        return error;
      }

      //! symmetric mode: start of an iteration after roundSymmetric: frame of y, right sets, boundaries, error crosses
      //!
      //! @return the largest error estimate on the error crosses (0 without enrichment)
      //!
      RealType prepareSymmetricSweep()
      {
        assert(sym_ && !sym_->yIsZero);
        setupSymmetricFrame(false, J_);
        updateBoundaries();
        // the error crosses inside the half from right to left, on top of the mirrored J^z_m
        RealType error = 0;
        if( enrich_ )
        {
          const RealType scale = rmsRef_ > 0 ? rmsRef_ : 1;
          for(int k = nCores_-1; k >= 1; k--)
            error = std::max(error, updateErrorCrossRight(k, scale));
        }
        return error;
      }

      //! symmetric mode: the symmetric two-site step at the middle bond, after stepLeftToRight for the cores 0..m-2
      StepMeasures middleStep()
      {
        assert(sym_);
        const int m = nCores_;
        const int n = dims_[m-1];

        // interface of the new left-orthonormal cores 0..m-2 of y on I_{m-2} (the 1x1 identity for m = 1)
        Ly_.update(m-2, m);
        const Tensor2<T> L = Ly_.left() ? interfaceLeft(Ly_.left()->get(), I_[m-2]) : boundaryInterface<T>();
        const int q = L.r1();

        // the full sets I_{m-2} x S_{m-1} (and I^z_{m-2} x S_{m-1}) at the middle bond, mirrored
        I_[m-1] = fullLeftIndexSet(q, n);
        const NestedIndices oldIz = enrich_ ? Iz_[m-1] : NestedIndices();
        const int qz = enrich_ && m > 1 ? Iz_[m-2].size() : 1;
        if( enrich_ )
          Iz_[m-1] = fullLeftIndexSet(qz, n);
        updateBoundaries();

        // f on both middle crosses (before I_{m-1} changes: the boundary matrices of x belong to the full sets)
        const Tensor3<T> B = fx_.bondBlock(m);
        Tensor3<T> residual = enrich_ ? fx_.errorBondBlock(m) : Tensor3<T>();
        rmsRef_ = std::max(rmsRef_, t3_rms(B));
        const RealType scale = rmsRef_ > 0 ? rmsRef_ : 1;

        // change: the previous y on this cross (zero in the first iteration)
        Tensor3<T> yOld(B.r1(), 1, B.r2());
        if( sym_->yIsZero )
          yOld.setConstant(T(0));
        else
          EigenMap(unfold_left(yOld)) = symmetricValues(I_);
        const RealType change = rmsDifference(B, yOld) / scale;

        // coordinates of the new left part (positive semidefinite up to round-off: a principal cross of the positive
        // semidefinite f(x), congruence), its truncated positive part: the new last core of G and lambda
        const Tensor2<T> C = solveLeftInterfaceSymmetric(L, unfold_left(B), n);
        PositiveEigen<T> eig = positiveEigen(C, localTolerance_, maxRanks_[m-1]);
        const RealType normC = ConstEigenMap(C).norm();
        sym_->negativePart = normC > 0 ? eig.negativeNorm / normC : RealType(0);
        Tensor3<T> core = fold_left(std::move(eig.U), n);
        I_[m-1] = leftIndicesFromRows(selectLeftRows(L, core, options_.maxvolTolerance, options_.oversampling, maxIndices_[m-1]), q);

        std::vector<Tensor3<T>> cores(m);
        for(int j = 0; j < m-1; j++)
          copy(TTy_.subTensor(j), cores[j]);
        cores[m-1] = std::move(core);
        sym_->G = TensorTrain<T>(std::move(cores), std::vector<TT_Orthogonality>(m, TT_Orthogonality::left));
        sym_->lambda = std::move(eig.lambda);
        sym_->yIsZero = false;

        // error crosses: I^z_{m-1} from the error f(x) - y on (I^z_{m-2} x S_{m-1}) x mirror
        RealType error = 0;
        if( enrich_ )
        {
          Tensor3<T> yNew(residual.r1(), 1, residual.r2());
          EigenMap(unfold_left(yNew)) = symmetricValues(Iz_);
          internal::t3_axpy(T(-1), yNew, residual);
          error = t3_rms(residual) / scale;
          if( keepErrorCross(oldIz, error, qz) )
            Iz_[m-1] = oldIz;
          else
          {
            Tensor2<T> R2(qz*n, qz*n);
            EigenMap(R2) = ConstEigenMap(unfold_left(residual));
            const Tensor3<T> rows = fold_left(std::move(R2), n);
            Iz_[m-1] = leftIndicesFromRows(dominantRows(rows, errorTolerance(rows, scale), options_.errorRank, options_.maxvolTolerance), qz);
          }
        }
        return {change, error};
      }

      //! symmetric mode: TT-rounding of y (truncates the enrichment and the ranks that are larger than needed)
      void roundSymmetric()
      {
        assert(sym_ && !sym_->yIsZero);
        // the same tolerance per bond as the local steps (symmetricRound uses rankTolerance / sqrt(d-1) per bond)
        symmetricRound(sym_->G, sym_->lambda, T(options_.rankTolerance / 2), options_.maxRank);
      }

      //! symmetric mode: the ranks of y at the bonds 0..m-1 (the last one is the middle bond)
      [[nodiscard]] std::vector<int> symmetricRanks() const
      {
        std::vector<int> ranks = sym_->G.getTTranks();
        ranks.push_back(sym_->G.subTensor(nCores_-1).r2());
        return ranks;
      }

      //! symmetric mode: store the result y = TTy TTy^T in TTy (TTy = G diag(sqrt(lambda)))
      void storeSymmetricResult()
      {
        assert(sym_);
        halfOfY(TTy_);
      }

      //! relative error of y on the control sample (0 without control sample), see TTCrossInfo::lastSampleError
      [[nodiscard]] RealType sampleError() const
      {
        if( sampleIndices_.rows() == 0 )
          return 0;
        Tensor3<T> yOnSample;
        if( sym_ )
        {
          TensorTrain<T> H(sym_->G.dimensions());
          halfOfY(H);
          evaluateAt(H, sampleIndices_, yOnSample, true);
        }
        else
          evaluateAt(TTy_, sampleIndices_, yOnSample);
        return rmsDifference(fOnSample_, yOnSample) / (rmsfOnSample_ > 0 ? rmsfOnSample_ : 1);
      }

      //! store the number of evaluations and the final index sets as full multi-indices in info
      void storeFinalInfo(TTCrossInfo<T>& info) const
      {
        info.nEvaluations = nEvaluations();
        if( sym_ )
        {
          info.lastNegativePart = sym_->negativePart;
          // the left sets of the half (including the middle bond) and their mirrors (the right sets inside the half come
          // from the frame of the last iteration and are not stored)
          info.leftIndices.resize(nCores_);
          for(int k = 0; k < nCores_; k++)
            info.leftIndices[k] = expandLeft(I_, k);
          std::vector<NestedIndices> J(nDim_);
          mirrorIndexSets(I_, nCores_, J);
          info.rightIndices.resize(nDim_);
          for(int k = nCores_; k < nDim_; k++)
            info.rightIndices[k] = expandRight(J, k, nDim_);
          return;
        }
        info.leftIndices.resize(std::max(nDim_-1, 0));
        for(int k = 0; k < nDim_-1; k++)
          info.leftIndices[k] = expandLeft(I_, k);
        info.rightIndices.resize(nDim_);
        for(int k = 1; k < nDim_; k++)
          info.rightIndices[k] = expandRight(J_, k, nDim_);
      }

    private:
      //! the options after checking them together with the dimensions of the tensors
      static const TTCrossOptions<T>& checkedOptions(const Inputs& TTx, const TensorTrain<T>& TTy, const TTCrossOptions<T>& options, bool symmetric)
      {
        const std::vector<int>& dims = TTy.dimensions();
        const int nDim = dims.size();
        if( nDim < 1 )
          throw std::invalid_argument("applyElementwise: empty tensor train!");
        for(const TensorTrain<T>& TT: TTx)
        {
          if( TT.dimensions() != dims )
            throw std::invalid_argument("applyElementwise: dimensions of TTx and TTy differ!");
          // (the halves of symmetric tensors have an open last bond)
          if( TT.subTensor(0).r1() != 1 || (!symmetric && TT.subTensor(nDim-1).r2() != 1) )
            throw std::invalid_argument("applyElementwise: boundary ranks of TTx must be 1!");
        }
        if( options.nMALS < 1 || options.nMALS > 2 )
          throw std::invalid_argument("applyElementwise: nMALS must be 1 or 2!");
        if( options.errorRank < 0 || options.enrichmentRank < 0 || options.enrichmentRank > options.errorRank )
          throw std::invalid_argument("applyElementwise: need 0 <= enrichmentRank <= errorRank!");
        if( options.nSweeps < 1 || options.maxRank < 1 )
          throw std::invalid_argument("applyElementwise: nSweeps and maxRank must be positive!");
        if( !(std::real(options.oversampling) >= 0) )
          throw std::invalid_argument("applyElementwise: oversampling must not be negative!");
        if( symmetric && options.randomErrorIndexSets )
          throw std::invalid_argument("applyElementwiseSymmetric: randomErrorIndexSets is not supported!");
        return options;
      }

      //! Limits per bond k (between cores k and k+1): a TT rank at bond k is at most n_k times the rank at bond k-1 and
      //! n_{k+1} times the rank at bond k+1, so a small bond also limits its neighbors (one pass in each direction suffices).
      static std::vector<int> limitByNeighbors(const std::vector<int>& dims, std::vector<long long> limit)
      {
        const int nDim = dims.size();
        for(int k = 0; k < nDim-1; k++)
          limit[k] = std::min(limit[k], dims[k] * (k == 0 ? 1 : limit[k-1]));
        for(int k = nDim-2; k >= 0; k--)
          limit[k] = std::min(limit[k], dims[k+1] * (k == nDim-2 ? 1 : limit[k+1]));
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

      //! dimensions of the symmetric tensor of a half: n_0, ..., n_{m-1}, n_{m-1}, ..., n_0
      static std::vector<int> mirroredDimensions(const std::vector<int>& halfDims)
      {
        std::vector<int> dims(halfDims);
        dims.insert(dims.end(), halfDims.rbegin(), halfDims.rend());
        return dims;
      }

      //! the first n entries of a vector (limits per bond for the bonds of the half)
      static std::vector<int> firstEntries(const std::vector<int>& v, int n)
      {
        return std::vector<int>(v.begin(), v.begin() + n);
      }

      //! symmetric mode: the boundary matrix of y on its own sets or on the error sets (nullopt: the 1x1 identity otherwise)
      [[nodiscard]] optional_cref<Tensor2<T>> yBoundary(bool errorSets) const
      {
        if( !sym_ )
          return std::nullopt;
        return std::cref(errorSets ? sym_->Ryz : sym_->Ry);
      }

      //! symmetric mode: the values of the half G on I_{m-1}, transposed (the boundary matrix of the mirrored G^T on J_m)
      [[nodiscard]] Tensor2<T> boundaryOf(const TensorTrain<T>& G, const std::vector<NestedIndices>& I) const
      {
        return transposedInterfaceLeft(leftPartialContraction(G, I, nCores_-1), I[nCores_-1]);
      }

      //! symmetric mode: y on the cross (I_{m-2} x S_{m-1}) x mirror (rows and columns in the row order of the left unfolding)
      [[nodiscard]] Eigen::MatrixX<T> symmetricValues(const std::vector<NestedIndices>& I) const
      {
        const Tensor3<T> LX = leftPartialContraction(sym_->G, I, nCores_-1);
        const auto A = ConstEigenMap(unfold_left(LX));
        const Eigen::Map<const Eigen::VectorX<T>> lambda(sym_->lambda.data(), sym_->lambda.size());
        return A * lambda.asDiagonal() * A.transpose();
      }

      //! symmetric mode: the half H = G diag(sqrt(lambda)) of y = H H^T
      void halfOfY(TensorTrain<T>& H) const
      {
        std::vector<T> sqrtLambda(sym_->lambda.size());
        for(int i = 0; i < std::ssize(sqrtLambda); i++)
          sqrtLambda[i] = std::sqrt(sym_->lambda[i]);
        scaleOpenBond(sym_->G, sqrtLambda, H);
      }

      //! symmetric mode: TTy = G diag(lambda) in the right-orthonormal form (zero values for y = 0 on the initial frame), the
      //! mirrored right sets J at and beyond the middle bond, the boundary matrix Ry and the right sets inside the half (maxvol)
      void setupSymmetricFrame(bool zeroValues, std::vector<NestedIndices>& J)
      {
        const int m = nCores_;
        rightOrthonormalHalf(sym_->G, sym_->lambda, TTy_);
        if( zeroValues )
        {
          Tensor3<T> zero(1, dims_[0], TTy_.subTensor(0).r2());
          zero.setConstant(T(0));
          TTy_.setSubTensor(0, std::move(zero));
        }
        mirrorIndexSets(I_, m, J);
        sym_->Ry = boundaryOf(sym_->G, I_);
        const std::vector<NestedIndices> Jhalf = rightIndexSetsFromFrame(TTy_, options_.maxvolTolerance, options_.oversampling,
                                                                         firstEntries(maxIndices_, m-1), yBoundary(false));
        for(int k = 1; k < m; k++)
          J[k] = Jhalf[k];
      }

      //! symmetric mode: mirror the left sets of y and of the error crosses, recompute the boundary matrices of x and of y
      //! on the error crosses (that of y on its own sets is part of setupSymmetricFrame), invalidate the contractions of y
      void updateBoundaries()
      {
        const int m = nCores_;
        mirrorIndexSets(I_, m, J_);
        if( enrich_ )
        {
          mirrorIndexSets(Iz_, m, Jz_);
          sym_->Ryz = boundaryOf(sym_->G, Iz_);
        }
        fx_.mirrorRightBoundary();
        Ly_.invalidate(0, m);
        Lyz_.invalidate(0, m);
      }

      //! initialize y with the right frame of the inputs and return the right index sets from this frame
      //!
      //! Symmetric: G diag(lambda) G^T is the frame of the inputs (initialHalfFrame), y = 0 on it, the left sets from a
      //! maxvol sweep over G and the right sets from the frame.
      //!
      std::vector<NestedIndices> initializeYAndRightIndexSets()
      {
        if( !sym_ )
        {
          initialRightFrame(inputs_, TTy_);
          return rightIndexSetsFromFrame(TTy_, options_.maxvolTolerance, options_.oversampling, maxIndices_);
        }
        initialHalfFrame(inputs_, sym_->G, sym_->lambda);
        const std::vector<NestedIndices> Ihalf = leftIndexSetsFromFrame(sym_->G, options_.maxvolTolerance, options_.oversampling,
                                                                        firstEntries(maxIndices_, nCores_), true);
        for(int k = 0; k < nCores_; k++)
          I_[k] = Ihalf[k];
        std::vector<NestedIndices> J(nDim_);
        setupSymmetricFrame(true, J);
        return J;
      }

      //! initial right index sets of the error crosses (empty without enrichment; symmetric: also sets Iz_)
      std::vector<NestedIndices> initialErrorIndexSets()
      {
        if( !enrich_ )
          return std::vector<NestedIndices>(nDim_);
        if( sym_ )
          return initialSymmetricErrorIndexSets();
        if( options_.randomErrorIndexSets )
          return randomRightIndexSets(dims_, options_.errorRank);
        // maxvol sets of a frame of rank errorRank of the inputs: well-conditioned crosses of the dominant part of x
        TensorTrain<T> TTz(dims_);
        initialRightFrame(inputs_, TTz, options_.errorRank);
        return rightIndexSetsFromFrame(TTz, options_.maxvolTolerance);
      }

      //! symmetric mode: initial error crosses from the dominant rank-errorRank part of the frame, with errorRank rows/columns
      //! where possible (fixed size as zrank in amen_cross: with only the rank of the frame at a bond, the error crosses would
      //! be the maxvol crosses of the same frame as those of y, where y interpolates exactly)
      std::vector<NestedIndices> initialSymmetricErrorIndexSets()
      {
        const int m = nCores_;
        TensorTrain<T> Gz(sym_->G.dimensions());
        copy(sym_->G, Gz);
        std::vector<T> lambdaz = sym_->lambda;
        symmetricRound(Gz, lambdaz, T(0), options_.errorRank);
        const T oversampleToErrorRank = T(options_.errorRank);
        const std::vector<NestedIndices> Ihalf = leftIndexSetsFromFrame(Gz, options_.maxvolTolerance, oversampleToErrorRank,
                                                                        std::vector<int>(m, options_.errorRank), true);
        for(int k = 0; k < m; k++)
          Iz_[k] = Ihalf[k];
        std::vector<NestedIndices> Jz(nDim_);
        mirrorIndexSets(Iz_, m, Jz);
        const Tensor2<T> Rz = boundaryOf(Gz, Iz_);
        TensorTrain<T> Z(Gz.dimensions());
        rightOrthonormalHalf(Gz, lambdaz, Z);
        const std::vector<NestedIndices> Jhalf = rightIndexSetsFromFrame(Z, options_.maxvolTolerance, oversampleToErrorRank,
                                                                         std::vector<int>(m-1, options_.errorRank), std::optional(std::cref(Rz)));
        for(int k = 1; k < m; k++)
          Jz[k] = Jhalf[k];
        return Jz;
      }

      //! The interface matrices of y at the cross (I_{k-1}, S_k x ... x S_{k+nMALS-1}, J_{k+nMALS}) of the block of the
      //! cores k..k+nMALS-1 (the 1x1 identity at the boundary, symmetric: Ry at the right end of the half; needs
      //! Ly.update(k-1, k+nMALS) first)
      std::pair<Tensor2<T>, Tensor2<T>> interfacesOfY(int k) const
      {
        Tensor2<T> R;
        if( Ly_.right() )
          R = interfaceRight(Ly_.right()->get(), J_[k+nMALS_]);
        else if( sym_ )
          copy(sym_->Ry, R);   // the half ends with the mirrored G^T
        else
          R = boundaryInterface<T>();
        return {Ly_.left() ? interfaceLeft(Ly_.left()->get(), I_[k-1]) : boundaryInterface<T>(), std::move(R)};
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

      //! symmetric mode: y = G diag(lambda) G^T (see the symmetric variant in pitts_tensortrain_cross.hpp)
      struct SymmetricPart
      {
        explicit SymmetricPart(const std::vector<int>& halfDims) : G(halfDims) {}
        //! left-orthonormal half (between the iterations; the fixed mirrored right part during a sweep)
        TensorTrain<T> G;
        //! weights lambda > 0 (a single 0 if no eigenvalue was positive; before the first middle step those of the initial frame)
        std::vector<T> lambda;
        //! y is still zero (before the first middle step)
        bool yIsZero = true;
        //! boundary matrices of y: the values of the mirrored G^T on J_m and on J^z_m
        Tensor2<T> Ry;
        Tensor2<T> Ryz;
        //! norm of the negative eigenvalues removed in the last middle step, relative to the norm of its block C (see TTCrossInfo)
        RealType negativePart = 0;
      };

      // (the order of the members matters, see the constructor)
      const TTCrossOptions<T>& options_;
      //! number of cores of TTy (the half in symmetric mode)
      const int nCores_;
      //! dimensions of the tensor (symmetric: of the full tensor, 2 nCores)
      const std::vector<int> dims_;
      const int nDim_;
      const int nMALS_;
      const bool enrich_;
      //! truncation tolerance of a local step (relative to the block, per bond of the full tensor)
      const T localTolerance_;
      const std::vector<int> maxRanks_;
      const std::vector<int> maxIndices_;
      const std::vector<std::reference_wrapper<const TensorTrain<T>>> inputs_;
      TensorTrain<T>& TTy_;
      //! index sets of y: I[k] for k = 0..nDim-2, J[k] for k = 1..nDim-1 (J[0] unused); symmetric: I[k] for k = 0..m-1
      std::vector<NestedIndices> I_;
      //! the state of the symmetric mode (empty otherwise)
      std::optional<SymmetricPart> sym_;
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

    state.storeFinalInfo(info);
    return info;
  }


  //! Symmetric TT-cross for y ~ f(x_1, ..., x_N) with x_i = H_i H_i^T given by the halves TTx, see applyElementwiseSymmetric
  template<typename T, typename Function, std::size_t N>
  TTCrossInfo<T> symmetricCrossSweeps(const Function& f, const std::array<std::reference_wrapper<const TensorTrain<T>>, N>& TTx,
                                      TensorTrain<T>& TTy, const TTCrossOptions<T>& options)
  {
    static_assert(std::is_floating_point_v<T>, "applyElementwiseSymmetric: only for real types (symmetric eigenvalue problem)");
    using RealType = typename CrossSweepState<T, Function, N>::RealType;

    CrossSweepState<T, Function, N> state(f, TTx, TTy, options, true);
    TTCrossInfo<T> info;

    const auto printProgress = [&](int sweep, RealType change, RealType error)
    {
      std::cout << "symmetric TT-cross sweep " << sweep << ": change " << change << ", error estimate " << error;
      if( options.controlSampleSize > 0 )
        std::cout << ", sampled error " << info.lastSampleError;
      std::cout << ", ranks of the half (the last one at the middle bond):";
      for(int r: state.symmetricRanks())
        std::cout << " " << r;
      std::cout << "\n";
    };

    const int m = state.nCores();
    const SweepIndex firstIdx(m, state.nMALS(), state.nMALS()-1, 0);
    // rank growth as a second convergence criterion, on the ranks after the rounding (see crossSweeps)
    const auto maxRankOf = [](const std::vector<int>& ranks){return ranks.empty() ? 0 : *std::max_element(ranks.begin(), ranks.end());};
    int largestRank = maxRankOf(state.symmetricRanks());
    for(int iSweep = 0; iSweep < options.nSweeps; iSweep++)
    {
      // the first iteration starts with the initial frame (y = 0, its change measure is meaningless)
      RealType maxChange = 0, maxError = 0;
      if( iSweep > 0 )
        maxError = state.prepareSymmetricSweep();
      // left to right up to the middle (the cores 0..m-2 as left cores of the steps), then the middle bond
      for(SweepIndex swpIdx = firstIdx.first(); swpIdx && swpIdx.leftDim() <= m-2; swpIdx = swpIdx.next())
      {
        const auto [change, error] = state.stepLeftToRight(swpIdx);
        maxChange = std::max(maxChange, change);
        maxError = std::max(maxError, error);
      }
      const auto [change, error] = state.middleStep();
      maxChange = std::max(maxChange, change);
      maxError = std::max(maxError, error);
      state.roundSymmetric();

      info.nSweeps = iSweep+1;
      info.lastChange = maxChange;
      info.lastError = maxError;
      info.lastSampleError = state.sampleError();
      if( options.verbose )
        printProgress(iSweep+1, maxChange, maxError);

      const int previousRank = std::exchange(largestRank, maxRankOf(state.symmetricRanks()));
      if( maxChange < options.convergenceTolerance && maxError < options.convergenceTolerance && largestRank <= previousRank )
      {
        info.converged = true;
        break;
      }
    }

    state.storeFinalInfo(info);
    state.storeSymmetricResult();
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

  // implement applyElementwiseSymmetric
  template<typename T, typename Function>
  TTCrossInfo<T> applyElementwiseSymmetric(const Function& f, const TensorTrain<T>& TTx, TensorTrain<T>& TTy, const TTCrossOptions<T>& options)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    return internal::tt_cross::symmetricCrossSweeps<T, Function, 1>(f, {std::cref(TTx)}, TTy, options);
  }

  // implement applyElementwiseSymmetric for two inputs
  template<typename T, typename Function>
  TTCrossInfo<T> applyElementwiseSymmetric(const Function& f, const TensorTrain<T>& TTx1, const TensorTrain<T>& TTx2, TensorTrain<T>& TTy,
                                           const TTCrossOptions<T>& options)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    return internal::tt_cross::symmetricCrossSweeps<T, Function, 2>(f, {std::cref(TTx1), std::cref(TTx2)}, TTy, options);
  }
}


#endif // PITTS_TENSORTRAIN_CROSS_IMPL_HPP
