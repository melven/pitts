// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_cross_helper_impl.hpp
* @brief Building blocks for TT-cross algorithms: nested index sets, interface matrices, core updates
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-11
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_CROSS_HELPER_IMPL_HPP
#define PITTS_TENSORTRAIN_CROSS_HELPER_IMPL_HPP

// includes
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include "pitts_tensortrain_cross_helper.hpp"
#include "pitts_tensor2_concat.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor2_maxvol.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensortrain_normalize.hpp"
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
      // implement enrichFactors
      //
      // Directions inside the span of Q are dropped, and never more than the dimension of the orthogonal
      // complement of Q is added (rho' <= min(rho, r1 n - r)); with a rank-deficient remainder the QR below
      // would otherwise return noise directions, and the maxvol that follows fails because its matrix is
      // not tall.
      //
      // Same sequence as the enrichSubspace lambda of solveMALS: project out, rank-revealing QR of the
      // remainder with an absolute tolerance, concatenate with zero padding on the unfoldings. Only needed
      // for the left-to-right sweeps, which are the only ones that enrich (pitts_tensortrain_cross.hpp).
      template<typename T>
      void enrichFactors(Tensor3<T>& Q, Tensor2<T>& B, Tensor3<T>&& E, int maxDirections, T relativeTolerance)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        using RealType = decltype(std::abs(T()));

        if( Q.r1() != E.r1() || Q.n() != E.n() )
          throw std::invalid_argument("TT-cross enrichment: dimension mismatch!");

        const int rho = std::min<long long>({E.r2(), maxDirections, Q.r1()*Q.n() - Q.r2()});
        if( rho <= 0 )
          return;

        // Absolute tolerance relative to the size of E: a direction whose component outside span(Q) is below
        // relativeTolerance * ||E|| is dropped. It would not enlarge the space in any useful way (the block SVD
        // truncates at a similar tolerance), and the projection remainder of the factors from
        // normalize_svd/normalize_qb is well above eps * ||E|| (observed ~1e-12 for double).
        // Computed before the projection overwrites E.
        const RealType tolerance = std::real(relativeTolerance) * t3_nrm(E);

        // Project out the known directions, in place: E := E - Q (Q^H Q)^{-1} (Q^H E).
        // Q is only orthogonal up to round-off (internal::normalize_svd does not compute its orthogonal
        // factor but reconstructs it as Q = M V S^{-1} from the Q-less TSQR, so ||Q^H Q - I|| grows like
        // eps*sigma_1/sigma_r). With the plain projection E := E - Q (Q^H E) the result keeps a component
        // of that size in the span of Q; near convergence E lies almost completely in span(Q), the
        // subtraction cancels nearly everything and this component dominates what is left. The enriched
        // factor then loses its orthogonality (max|Q^H Q - I| = O(1) observed) and the maxvol in
        // updateCoreLeft aborts with a rank-deficient matrix. Correcting with the Gram matrix of Q (which
        // is almost the identity) makes this the exact projection onto the orthogonal complement of Q,
        // independent of the orthogonality error -- the same fix as in the residual calculation of
        // PITTS::solveMALS.
        Eigen::MatrixX<T> QtQ = ConstEigenMap(unfold_left(Q)).adjoint() * ConstEigenMap(unfold_left(Q));
        const Eigen::LLT<Eigen::Ref<Eigen::MatrixX<T>>> llt(QtQ);   // in-place decomposition
        const auto projectOut = [&](Tensor3<T>& X)
        {
          Tensor2<T> QtX;
          reverse_dot_contract2(Q, X, QtX);     // QtX = Q(*,*,:) * X(*,*,:)
          if( llt.info() == Eigen::Success )
            llt.solveInPlace(EigenMap(QtX));
          Tensor3<T> QQtX;
          normalize_contract2(Q, QtX, QQtX);    // QQtX = Q(:,:,*) * QtX(*,:)
          t3_axpy(T(-1), QQtX, X);
        };
        projectOut(E);

        if( t3_nrm(E) <= tolerance )
          return;

        // More candidates than wanted (as in solveMALS): keep the dominant directions of the projected E (SVD),
        // scaled by their singular values, and project them once more (the SVD only reconstructs its orthogonal
        // factor, see above). The QR below then orthonormalizes them.
        if( rho < E.r2() )
        {
          auto [QE, SVt] = normalize_svd(unfold_left(E), true, T(tolerance), rho, true);
          for(int j = 0; j < QE.r2(); j++)
            EigenMap(QE).col(j) *= ConstEigenMap(SVt).row(j).norm();
          E = fold_left(std::move(QE), Q.n());
          projectOut(E);
        }

        const auto [QE, RE] = normalize_qb(unfold_left(E), true, T(tolerance), rho, true);
        if( QE.r2() == 0 )
          return;

        Tensor3<T> Qext(Q.r1(), Q.n(), Q.r2() + QE.r2());
        concatLeftRight<T>(unfold_left(Q), ConstTensor2View<T>(QE), unfold_left(Qext));
        Tensor2<T> Bext(B.r1() + QE.r2(), B.r2());
        concatTopBottom<T>(ConstTensor2View<T>(B), std::nullopt, Bext);
        Q = std::move(Qext);
        B = std::move(Bext);
      }


      // implement maxvolColumns
      template<typename T>
      std::vector<int> maxvolColumns(const ConstTensor2View<T>& Vt, T maxvolTolerance, Tensor2<T>* coefficients, int nColumns)
      {
        const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

        Tensor2<T> VtH(Vt.r2(), Vt.r1());
        EigenMap(VtH) = ConstEigenMap(Vt).adjoint();

        Tensor2<T> coefficientsH;
        const std::vector<int> columns = maxvol(VtH, maxvolTolerance, 100, coefficients ? &coefficientsH : nullptr, nColumns);

        if( coefficients )
        {
          // (Vt^H Vt^H(columns,:)^+)^H = Vt(:,columns)^+ Vt (the inverse without oversampling)
          coefficients->resize(coefficientsH.r2(), coefficientsH.r1());
          EigenMap(*coefficients) = ConstEigenMap(coefficientsH).adjoint();
        }
        return columns;
      }


      // implement updateCoreLeft
      template<typename T>
      CoreUpdate<T> updateCoreLeft(const Tensor3<T>& B, T rankTolerance, int maxRank, T maxvolTolerance,
                                   Tensor3<T> enrichment, T oversampling, int maxEnrichment, int maxEnrichedRank, int maxRows)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int r1 = B.r1();
        const int n = B.n();
        const int r2 = B.r2();
        if( r1*n*r2 == 0 )
          throw std::invalid_argument("updateCoreLeft: empty block!");

        CoreUpdate<T> result;

        // 1. truncated SVD of the left unfolding: B ~ U * SVt with orthonormal columns of U
        //    (truncation in the Frobenius norm as in PITTS::normalize: the discarded part is bounded by
        //     rankTolerance * ||B||_F, whereas a threshold on the individual singular values would allow
        //     an error of sqrt(#discarded) * rankTolerance * sigma_1)
        auto [U, SVt] = normalize_svd(unfold_left(B), true, rankTolerance, maxRank, false, true);
        result.truncatedRank = U.r2();


        // 2. enrichment: append new orthonormal directions to U and zero rows to SVt
        //    (this does not change U*SVt, it only enlarges the space the next core can use;
        //     the projection works on the core-shaped factors, folding and unfolding are moves)
        if( enrichment.r2() > 0 )
        {
          Tensor3<T> U3 = fold_left(std::move(U), n);
          // directions below the truncation tolerance of the block are irrelevant (rankTolerance 0: only drop noise)
          using RealType = decltype(std::abs(T()));
          const RealType sqrtEps = std::sqrt(std::numeric_limits<RealType>::epsilon());
          const RealType relTol = std::real(rankTolerance) > 0 ? std::min(sqrtEps, RealType(std::real(rankTolerance))) : sqrtEps;
          enrichFactors(U3, SVt, std::move(enrichment), std::min<long long>(maxEnrichment, maxEnrichedRank - U3.r2()), T(relTol));
          U = unfold_left(std::move(U3));
        }
        const int r = U.r2();

        // 3.+4. maxvol selects the rows I_k (with oversampling more than r) and returns the core in interpolation
        //       gauge, U * U(rows,:)^{-1} (U * U(rows,:)^+ with oversampling)
        const int nRows = std::max(r, std::min(r + static_cast<int>(std::ceil(std::real(oversampling) * r)), maxRows));
        Tensor2<T> coefficients;
        result.selected = maxvol(U, maxvolTolerance, 100, &coefficients, nRows);
        result.core = fold_left(std::move(coefficients), n);

        // carry = U(rows,:) * SVt, so that core * carry = U * SVt
        // (the row selection as an explicit loop: an Eigen indexed view triggers -Warray-bounds false positives with GCC 13)
        const int q = result.selected.size();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Uhat(q, r);
        for(int j = 0; j < q; j++)
          Uhat.row(j) = ConstEigenMap(U).row(result.selected[j]);
        result.carry.resize(q, r2);
        EigenMap(result.carry).noalias() = Uhat * ConstEigenMap(SVt);

        return result;
      }


      // implement updateCoreRight
      template<typename T>
      CoreUpdate<T> updateCoreRight(const Tensor3<T>& B, T rankTolerance, int maxRank, T maxvolTolerance, T oversampling, int maxColumns)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int r1 = B.r1();
        const int n = B.n();
        const int r2 = B.r2();
        if( r1*n*r2 == 0 )
          throw std::invalid_argument("updateCoreRight: empty block!");

        CoreUpdate<T> result;

        // 1. truncated SVD of the right unfolding: B ~ US * Vt with orthonormal rows of Vt
        //    (truncation in the Frobenius norm, see updateCoreLeft)
        const auto [US, Vt] = normalize_svd(unfold_right(B), false, rankTolerance, maxRank, false, true);
        const int r = Vt.r1();
        result.truncatedRank = r;

        // 2. maxvol selects the columns J_k of Vt (with oversampling more than r) and returns the core in
        //    interpolation gauge, Vt(:,cols)^{-1} Vt (Vt(:,cols)^+ Vt with oversampling)
        const int nColumns = std::max(r, std::min(r + static_cast<int>(std::ceil(std::real(oversampling) * r)), maxColumns));
        Tensor2<T> coefficients;
        result.selected = maxvolColumns<T>(Vt, maxvolTolerance, &coefficients, nColumns);
        result.core = fold_right(std::move(coefficients), n);

        // carry = US * Vt(:,cols), so that carry * core = US * Vt (explicit loop, see updateCoreLeft)
        const int q = result.selected.size();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Vhat(r, q);
        for(int j = 0; j < q; j++)
          Vhat.col(j) = ConstEigenMap(Vt).col(result.selected[j]);
        result.carry.resize(r1, q);
        EigenMap(result.carry).noalias() = ConstEigenMap(US) * Vhat;

        return result;
      }


      // implement initialRightIndexSets
      template<typename T>
      std::vector<NestedIndices> initialRightIndexSets(const std::vector<std::reference_wrapper<const TensorTrain<T>>>& TTx,
                                                       T maxvolTolerance, int maxRank)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        if( TTx.empty() )
          throw std::invalid_argument("initialRightIndexSets: no tensor train!");
        if( maxRank < 1 )
          throw std::invalid_argument("initialRightIndexSets: maxRank must be positive!");
        const std::vector<int>& dims = TTx.front().get().dimensions();
        const int nDim = dims.size();
        for(const TensorTrain<T>& TT: TTx)
        {
          if( TT.dimensions() != dims )
            throw std::invalid_argument("initialRightIndexSets: dimensions of the tensor trains differ!");
          if( nDim < 1 || TT.subTensor(0).r1() != 1 || TT.subTensor(nDim-1).r2() != 1 )
            throw std::invalid_argument("initialRightIndexSets: boundary ranks must be 1!");
        }

        // Work on a right-orthogonalized copy: maxvol needs full column rank, and the right unfolding of a
        // right-orthogonal core has orthonormal rows, so X_k * R_{k+1} keeps full row rank as long as R_{k+1}
        // is a nonsingular (maxvol) submatrix. The orthogonalization also removes exactly dependent ranks
        // and makes the greedy initial pivoting of maxvol well-conditioned (the QR step of Oseledets/Tyrtyshnikov
        // 2010, reference [1] in pitts_tensortrain_cross.hpp).
        // The resulting sets are also valid for the original TTx: the interpolation entries x(I, S_k, J)
        // do not depend on the gauge of the cores.
        // A right sweep alone does not reveal rank deficiencies that sit in the left unfoldings (e.g. a zero
        // slice in X_0), so the copy is first left-normalized (minimal ranks) and then right-normalized.
        std::vector<TensorTrain<T>> TTortho;
        TTortho.reserve(TTx.size());    // no reallocation: the sweep data below refers to the elements
        for(const TensorTrain<T>& TT: TTx)
        {
          copy(TT, TTortho.emplace_back(dims));
          leftNormalize(TTortho.back(), T(0));
          rightNormalize(TTortho.back(), T(0));
        }

        // right-to-left sweep of the partial contractions XR_k = X_k R_{k+1} (one per input, all on the same
        // sets J); J[k+1] is set before XR_k is computed, so the same recursion (and boundary handling) as in
        // the cross sweeps can be used
        std::vector<NestedIndices> J(nDim);
        const auto neverFromTheLeft = [](int, optional_cref<Tensor3<T>>, Tensor3<T>&) { assert(false); };
        using RightContractions = decltype(defineSweepData<Tensor3<T>>(nDim, neverFromTheLeft, contraction_loop_from_right<T>(TTortho.front(), J)));
        std::vector<RightContractions> Rx;
        for(const TensorTrain<T>& TT: TTortho)
          Rx.push_back(defineSweepData<Tensor3<T>>(nDim, neverFromTheLeft, contraction_loop_from_right<T>(TT, J)));

        for(int k = nDim-1; k >= 1; k--)
        {
          int nRows = 0;
          for(auto& R: Rx)
          {
            R.update(-1, k);
            nRows += R.subTensor(k).r1();
          }

          // single input: the right unfolding (r_{k-1} x n |J_{k+1}|) is wide with full row rank,
          // maxvolColumns selects the columns J_k
          if( Rx.size() == 1 && nRows <= maxRank )
          {
            J[k] = rightIndicesFromColumns(maxvolColumns<T>(unfold_right(Rx.front().subTensor(k)), maxvolTolerance), dims[k]);
            continue;
          }

          // several inputs: the stacked right unfoldings may be rank-deficient or even tall, so select the
          // columns in an orthonormal basis of their joint row space
          Tensor2<T> stacked(nRows, Rx.front().subTensor(k).n() * Rx.front().subTensor(k).r2());
          int offset = 0;
          for(const auto& R: Rx)
          {
            const auto XR = unfold_right(R.subTensor(k));
            EigenMap(stacked).middleRows(offset, XR.r1()) = ConstEigenMap(XR);
            offset += XR.r1();
          }
          const Tensor2<T> Vt = normalize_svd(stacked, false, T(0), maxRank).second;
          J[k] = rightIndicesFromColumns(maxvolColumns<T>(Vt, maxvolTolerance), dims[k]);
        }

        return J;
      }
    }
  }
}


#endif // PITTS_TENSORTRAIN_CROSS_HELPER_IMPL_HPP
