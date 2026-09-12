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
      //! enrich the two factors of a truncated SVD by new directions of E, leaving their product unchanged
      //!
      //! The orthogonal factor Q is the core-shaped form (r1 x n x r) of a left-orthogonal matrix, B (r x l)
      //! the remainder, so unfold_left(Q)*B is the truncated block, and E (r1 x n x rho) holds candidate
      //! directions: the part of E outside the span of Q is orthonormalized and appended to Q, and B is
      //! padded with rho' zero rows.
      //! Directions inside the span of Q are dropped, and never more than the dimension of the orthogonal
      //! complement of Q is added (rho' <= min(rho, r1 n - r)); with a rank-deficient remainder the QR below
      //! would otherwise return noise directions, and the maxvol that follows fails because its matrix is
      //! not tall.
      //!
      //! Same sequence as the enrichSubspace lambda of solveMALS: project out, rank-revealing QR of the
      //! remainder with an absolute tolerance, concatenate with zero padding on the unfoldings. Only needed
      //! for the left-to-right sweeps, which are the only ones that enrich (pitts_tensortrain_cross.hpp).
      //!
      //! @param Q   orthogonal factor as a core, extended in place
      //! @param B   remainder, padded with zero rows in place
      //! @param E   candidate directions as a core
      //!
      template<typename T>
      void enrichFactors(Tensor3<T>& Q, Tensor2<T>& B, const Tensor3<T>& E)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        using RealType = decltype(std::abs(T()));

        if( Q.r1() != E.r1() || Q.n() != E.n() )
          throw std::invalid_argument("TT-cross enrichment: dimension mismatch!");

        const int rho = std::min<int>(E.r2(), Q.r1()*Q.n() - Q.r2());
        if( rho <= 0 )
          return;

        // project out the known directions: Ep = E - Q (Q^T E) (one pass is enough: Q is orthonormal up to
        // round-off, and small remaining components are removed by the tolerance below)
        Tensor2<T> QtE;
        reverse_dot_contract2(Q, E, QtE);       // QtE = Q(*,*,:) * E(*,*,:)
        Tensor3<T> QQtE;
        normalize_contract2(Q, QtE, QQtE);      // QQtE = Q(:,:,*) * QtE(*,:)
        Tensor3<T> Ep;
        copy(E, Ep);
        t3_axpy(T(-1), QQtE, Ep);

        // Absolute tolerance relative to the size of E: a direction whose component outside span(Q) is below
        // sqrt(eps) * ||E|| is dropped. It would not enlarge the space in any useful way (the block SVD
        // truncates at rankTolerance >= sqrt(eps) by default anyway), and the projection remainder of the
        // factors from normalize_svd/normalize_qb is well above eps * ||E|| (observed ~1e-12 for double).
        const RealType tolerance = std::sqrt(std::numeric_limits<RealType>::epsilon()) * t3_nrm(E);
        if( t3_nrm(Ep) <= tolerance )
          return;

        const auto [QE, RE] = normalize_qb(unfold_left(Ep), true, T(tolerance), rho, true);
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
      std::vector<int> maxvolColumns(const ConstTensor2View<T>& Vt, T maxvolTolerance, Tensor2<T>* coefficients)
      {
        const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

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
                                   optional_cref<Tensor3<T>> enrichment)
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
        //    (this does not change U*SVt, it only enlarges the space the next core can use;
        //     the projection works on the core-shaped factors, folding and unfolding are moves)
        if( enrichment )
        {
          Tensor3<T> U3 = fold_left(std::move(U), n);
          enrichFactors(U3, SVt, enrichment->get());
          U = unfold_left(std::move(U3));
        }
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
      CoreUpdate<T> updateCoreRight(const Tensor3<T>& B, T rankTolerance, int maxRank, T maxvolTolerance)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int r1 = B.r1();
        const int n = B.n();
        const int r2 = B.r2();
        if( r1*n*r2 == 0 )
          throw std::invalid_argument("updateCoreRight: empty block!");

        CoreUpdate<T> result;

        // 1. truncated SVD of the right unfolding: B ~ US * Vt with orthonormal rows of Vt
        const auto [US, Vt] = normalize_svd(unfold_right(B), false, rankTolerance, maxRank);
        const int r = Vt.r1();
        result.truncatedRank = r;

        // 2. maxvol selects the columns J_k of Vt and returns the core in interpolation gauge, Vt(:,cols)^{-1} Vt
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
