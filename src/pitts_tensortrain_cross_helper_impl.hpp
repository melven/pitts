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
#include "pitts_tensor2_maxvol.hpp"
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
      //! enrich the two factors of a truncated SVD by new directions of E, leaving their product unchanged
      //!
      //! For leftOrthog, Q (m x r) has orthonormal columns and B (r x l) the remainder, Q*B is the truncated
      //! block, and the columns of E (m x rho) are candidate directions: the part of E outside the span of Q is
      //! orthonormalized and appended to Q, B is padded with rho' zero rows. Mirrored for the transposed case.
      //! Directions inside the span of Q are dropped, and never more than the dimension of the orthogonal
      //! complement of Q is added (rho' <= min(rho, m-r)); with a rank-deficient remainder the QR below would
      //! otherwise return noise directions, and the maxvol that follows fails because its matrix is not tall.
      //!
      //! Same sequence as the enrichSubspace lambda of solveMALS: project out, rank-revealing QR of the
      //! remainder with an absolute tolerance, concatenate with zero padding.
      //!
      //! @param Q   orthogonal factor, extended in place
      //! @param B   remainder, padded with zeros in place
      //! @param E   candidate directions
      //! @param leftOrthog  true: columns of Q, Q*B with B below; false: rows of Q, B*Q with B to the left
      //!
      template<typename T>
      void enrichFactors(Tensor2<T>& Q, Tensor2<T>& B, const Tensor2<T>& E, bool leftOrthog)
      {
        using RealType = decltype(std::abs(T()));

        const auto mapQ = ConstEigenMap(Q);
        const auto mapE = ConstEigenMap(E);
        if( leftOrthog ? mapE.rows() != mapQ.rows() : mapE.cols() != mapQ.cols() )
          throw std::invalid_argument("TT-cross enrichment: dimension mismatch!");

        const int m = leftOrthog ? Q.r1() : Q.r2();
        const int r = leftOrthog ? Q.r2() : Q.r1();
        const int rho = std::min<int>(leftOrthog ? E.r2() : E.r1(), m - r);
        if( rho <= 0 )
          return;

        // project out the known directions (one pass is enough: Q is orthonormal up to round-off,
        // and small remaining components are removed by the tolerance below)
        Tensor2<T> Ep(E.r1(), E.r2());
        if( leftOrthog )
          EigenMap(Ep).noalias() = mapE - mapQ * (mapQ.adjoint() * mapE);
        else
          EigenMap(Ep).noalias() = mapE - (mapE * mapQ.adjoint()) * mapQ;

        // Absolute tolerance relative to the size of E: a direction whose component outside span(Q) is below
        // sqrt(eps) * ||E|| is dropped. It would not enlarge the space in any useful way (the block SVD
        // truncates at rankTolerance >= sqrt(eps) by default anyway), and the projection remainder of the
        // factors from normalize_svd/normalize_qb is well above eps * ||E|| (observed ~1e-12 for double).
        const RealType tolerance = std::sqrt(std::numeric_limits<RealType>::epsilon()) * mapE.norm();
        if( ConstEigenMap(Ep).norm() <= tolerance )
          return;

        auto [QE, RE] = normalize_qb(Ep, leftOrthog, T(tolerance), rho, true);
        const Tensor2<T>& newDirections = leftOrthog ? QE : RE;
        const int rhoNew = leftOrthog ? newDirections.r2() : newDirections.r1();
        if( rhoNew == 0 )
          return;

        Tensor2<T> Qext(leftOrthog ? Q.r1() : Q.r1() + rhoNew, leftOrthog ? Q.r2() + rhoNew : Q.r2());
        Tensor2<T> Bext(leftOrthog ? B.r1() + rhoNew : B.r1(), leftOrthog ? B.r2() : B.r2() + rhoNew);
        if( leftOrthog )
        {
          concatLeftRight<T>(Q, newDirections, Qext);
          concatTopBottom<T>(B, std::nullopt, Bext);
        }
        else
        {
          concatTopBottom<T>(Q, newDirections, Qext);
          concatLeftRight<T>(B, std::nullopt, Bext);
        }
        Q = std::move(Qext);
        B = std::move(Bext);
      }


      // implement maxvolColumns
      template<typename T>
      std::vector<int> maxvolColumns(const ConstTensor2View<T>& Vt, T maxvolTolerance, Tensor2<T>* coefficients)
      {
        Tensor2<T> VtH(Vt.r2(), Vt.r1());
        EigenMap(VtH) = ConstEigenMap(Vt).adjoint();

        Tensor2<T> coefficientsH;
        const std::vector<int> columns = maxvol(VtH, maxvolTolerance, 100, coefficients ? &coefficientsH : nullptr);

        if( coefficients )
        {
          // (Vt^H(columns,:)^{-1} Vt^H)^H = Vt(:,columns)^{-1} Vt
          coefficients->resize(Vt.r1(), Vt.r2());
          EigenMap(*coefficients) = ConstEigenMap(coefficientsH).adjoint();
        }
        return columns;
      }


      // implement updateCoreLeft
      template<typename T>
      CoreUpdate<T> updateCoreLeft(const Tensor3<T>& B, T rankTolerance, int maxRank, T maxvolTolerance,
                                   optional_cref<Tensor2<T>> enrichment)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int r1 = B.r1();
        const int n = B.n();
        const int r2 = B.r2();
        if( r1*n*r2 == 0 )
          throw std::invalid_argument("updateCoreLeft: empty block!");

        CoreUpdate<T> result;

        // 1. truncated SVD of the left unfolding: B ~ U * SVt with orthonormal columns of U
        auto [U, SVt] = normalize_svd(unfold_left(B), true, rankTolerance, maxRank);
        result.truncatedRank = U.r2();

        // 2. enrichment: append new orthonormal directions to U and zero rows to SVt
        //    (this does not change U*SVt, it only enlarges the space the next core can use)
        if( enrichment )
          enrichFactors(U, SVt, enrichment->get(), true);
        const int r = U.r2();

        // 3.+4. maxvol selects the rows I_k and returns the core in interpolation gauge, U * U(rows,:)^{-1}
        Tensor2<T> coefficients;
        result.selected = maxvol(U, maxvolTolerance, 100, &coefficients);
        result.core = fold_left(std::move(coefficients), n);

        // carry = U(rows,:) * SVt, so that core * carry = U * SVt
        // (the row selection as an explicit loop: an Eigen indexed view triggers -Warray-bounds false positives with GCC 13)
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Uhat(r, r);
        for(int j = 0; j < r; j++)
          Uhat.row(j) = ConstEigenMap(U).row(result.selected[j]);
        result.carry.resize(r, r2);
        EigenMap(result.carry).noalias() = Uhat * ConstEigenMap(SVt);

        return result;
      }


      // implement updateCoreRight
      template<typename T>
      CoreUpdate<T> updateCoreRight(const Tensor3<T>& B, T rankTolerance, int maxRank, T maxvolTolerance,
                                    optional_cref<Tensor2<T>> enrichment)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int r1 = B.r1();
        const int n = B.n();
        const int r2 = B.r2();
        if( r1*n*r2 == 0 )
          throw std::invalid_argument("updateCoreRight: empty block!");

        CoreUpdate<T> result;

        // 1. truncated SVD of the right unfolding: B ~ US * Vt with orthonormal rows of Vt
        auto [US, Vt] = normalize_svd(unfold_right(B), false, rankTolerance, maxRank);
        result.truncatedRank = Vt.r1();

        // 2. enrichment: append new orthonormal rows to Vt and zero columns to US
        if( enrichment )
          enrichFactors(Vt, US, enrichment->get(), false);
        const int r = Vt.r1();

        // 3.+4. maxvol selects the columns J_k of Vt and returns the core in interpolation gauge, Vt(:,cols)^{-1} Vt
        Tensor2<T> coefficients;
        result.selected = maxvolColumns<T>(Vt, maxvolTolerance, &coefficients);
        result.core = fold_right(std::move(coefficients), n);

        // carry = US * Vt(:,cols), so that carry * core = US * Vt (explicit loop, see updateCoreLeft)
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Vhat(r, r);
        for(int j = 0; j < r; j++)
          Vhat.col(j) = ConstEigenMap(Vt).col(result.selected[j]);
        result.carry.resize(r1, r);
        EigenMap(result.carry).noalias() = ConstEigenMap(US) * Vhat;

        return result;
      }


      // implement initialRightIndexSets
      template<typename T>
      std::vector<NestedIndices> initialRightIndexSets(const TensorTrain<T>& TTx, T maxvolTolerance)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int nDim = TTx.dimensions().size();
        if( nDim < 1 || TTx.subTensor(0).r1() != 1 || TTx.subTensor(nDim-1).r2() != 1 )
          throw std::invalid_argument("initialRightIndexSets: boundary ranks must be 1!");

        // Work on a right-orthogonalized copy: maxvol needs full column rank, and the right unfolding of a
        // right-orthogonal core has orthonormal rows, so X_k * R_{k+1} keeps full row rank as long as R_{k+1}
        // is a nonsingular (maxvol) submatrix. The orthogonalization also removes exactly dependent ranks
        // and makes the greedy initial pivoting of maxvol well-conditioned (the QR step of Oseledets/Tyrtyshnikov
        // 2010, reference [1] in pitts_tensortrain_cross.hpp).
        // The resulting sets are also valid for the original TTx: the interpolation entries x(I, S_k, J)
        // do not depend on the gauge of the cores.
        // A right sweep alone does not reveal rank deficiencies that sit in the left unfoldings (e.g. a zero
        // slice in X_0), so the copy is first left-normalized (minimal ranks) and then right-normalized.
        TensorTrain<T> TTortho(TTx.dimensions());
        copy(TTx, TTortho);
        leftNormalize(TTortho, T(0));
        rightNormalize(TTortho, T(0));

        // right-to-left sweep of the partial contractions XR_k = X_k R_{k+1}; J[k+1] is set before XR_k is
        // computed, so the same recursion (and boundary handling) as in the cross sweeps can be used
        std::vector<NestedIndices> J(nDim);
        const auto neverFromTheLeft = [](int, optional_cref<Tensor3<T>>, Tensor3<T>&) { assert(false); };
        auto Rx = defineSweepData<Tensor3<T>>(nDim, neverFromTheLeft, contraction_loop_from_right<T>(TTortho, J));

        for(int k = nDim-1; k >= 1; k--)
        {
          Rx.update(-1, k);
          const Tensor3<T>& XR = Rx.subTensor(k);

          // the right unfolding (r_{k-1} x n |J_{k+1}|) is wide, maxvolColumns selects the columns J_k
          J[k] = rightIndicesFromColumns(maxvolColumns<T>(unfold_right(XR), maxvolTolerance), XR.n());
        }

        return J;
      }
    }
  }
}


#endif // PITTS_TENSORTRAIN_CROSS_HELPER_IMPL_HPP
