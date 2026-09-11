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
      //! helper: orthonormal directions of E not yet contained in the orthonormal columns (leftOrthog) or rows of Q
      //!
      //! Returns Q_E (m x rho' for leftOrthog, rho' x m otherwise) with rho' <= rho; directions that are
      //! (numerically) in the span of Q are dropped. Same sequence as in the enrichSubspace lambda of
      //! solveMALS: project out, then a rank-revealing QR of the remainder with the rank limited to the
      //! dimension of the orthogonal complement of Q.
      //!
      template<typename T>
      Tensor2<T> newDirections(const Tensor2<T>& Q, const Tensor2<T>& E, bool leftOrthog)
      {
        using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        using RealType = decltype(std::abs(T()));

        const auto mapQ = ConstEigenMap(Q);
        const auto mapE = ConstEigenMap(E);
        if( leftOrthog && mapE.rows() != mapQ.rows() )
          throw std::invalid_argument("TT-cross enrichment: wrong number of rows!");
        if( !leftOrthog && mapE.cols() != mapQ.cols() )
          throw std::invalid_argument("TT-cross enrichment: wrong number of columns!");

        // the complement of span(Q) has dimension m - r: never return more directions than that
        // (with a rank-deficient remainder the QR below only sees round-off there and would otherwise
        // happily return noise directions; the following maxvol then fails because the matrix is not tall)
        const int m = leftOrthog ? Q.r1() : Q.r2();
        const int r = leftOrthog ? Q.r2() : Q.r1();
        const int rho = std::min<int>(leftOrthog ? E.r2() : E.r1(), m - r);
        Tensor2<T> empty(leftOrthog ? m : 0, leftOrthog ? 0 : m);
        if( rho <= 0 )
          return empty;

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
        const RealType scaleE = mapE.norm();
        const RealType tolerance = std::sqrt(std::numeric_limits<RealType>::epsilon()) * scaleE;
        if( ConstEigenMap(Ep).norm() <= tolerance )
          return empty;

        auto [QE, RE] = normalize_qb(Ep, leftOrthog, T(tolerance), rho, true);
        return leftOrthog ? std::move(QE) : std::move(RE);
      }


      // implement updateCoreLeft
      template<typename T>
      CoreUpdate<T> updateCoreLeft(const Tensor3<T>& B, T rankTolerance, int maxRank, T maxvolTolerance,
                                   optional_cref<Tensor2<T>> enrichment)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();
        using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

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
        if( enrichment && enrichment->get().r2() > 0 )
        {
          const Tensor2<T> QE = newDirections(U, enrichment->get(), true);
          if( QE.r2() > 0 )
          {
            Tensor2<T> Uext(U.r1(), U.r2() + QE.r2());
            concatLeftRight<T>(U, QE, Uext);
            Tensor2<T> SVtext(SVt.r1() + QE.r2(), r2);
            concatTopBottom<T>(SVt, std::nullopt, SVtext);
            U = std::move(Uext);
            SVt = std::move(SVtext);
          }
        }
        const int r = U.r2();

        // 3.+4. maxvol selects the rows I_k and returns the core in interpolation gauge, U * U(rows,:)^{-1}
        Tensor2<T> coefficients;
        result.selected = maxvol(U, maxvolTolerance, 100, &coefficients);
        result.core = fold_left(std::move(coefficients), n);

        // carry = U(rows,:) * SVt, so that core * carry = U * SVt
        mat Uhat(r, r);
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
        using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

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
        if( enrichment && enrichment->get().r1() > 0 )
        {
          const Tensor2<T> QE = newDirections(Vt, enrichment->get(), false);
          if( QE.r1() > 0 )
          {
            Tensor2<T> Vtext(Vt.r1() + QE.r1(), Vt.r2());
            concatTopBottom<T>(Vt, QE, Vtext);
            Tensor2<T> USext(r1, US.r2() + QE.r1());
            concatLeftRight<T>(US, std::nullopt, USext);
            Vt = std::move(Vtext);
            US = std::move(USext);
          }
        }
        const int r = Vt.r1();

        // 3.+4. maxvol works on tall matrices, so apply it to Vt^H (n r2 x r): it selects the columns J_k of Vt
        //       and returns Vt^H * (Vt^H(cols,:))^{-1} = (Vt(:,cols)^{-1} * Vt)^H, the adjoint of the core in
        //       interpolation gauge
        Tensor2<T> VtH(Vt.r2(), r);
        EigenMap(VtH) = ConstEigenMap(Vt).adjoint();
        Tensor2<T> coefficientsH;
        result.selected = maxvol(VtH, maxvolTolerance, 100, &coefficientsH);
        Tensor2<T> coefficients(r, Vt.r2());
        EigenMap(coefficients) = ConstEigenMap(coefficientsH).adjoint();
        result.core = fold_right(std::move(coefficients), n);

        // carry = US * Vt(:,cols), so that carry * core = US * Vt
        mat Vhat(r, r);
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
        // and makes the greedy initial pivoting of maxvol well-conditioned (the QR trick of the TT-cross paper).
        // The resulting sets are also valid for the original TTx: the interpolation entries x(I, S_k, J)
        // do not depend on the gauge of the cores.
        // A right sweep alone does not reveal rank deficiencies that sit in the left unfoldings (e.g. a zero
        // slice in X_0), so the copy is first left-normalized (minimal ranks) and then right-normalized.
        TensorTrain<T> TTortho(TTx.dimensions());
        copy(TTx, TTortho);
        leftNormalize(TTortho, T(0));
        rightNormalize(TTortho, T(0));

        std::vector<NestedIndices> J(nDim);
        Tensor2<T> R;     // right interface R_{k+1} of TTortho on J_{k+1}, empty for k = nDim-1 (R_d = 1x1 identity)
        for(int k = nDim-1; k >= 1; k--)
        {
          const Tensor3<T>& X = TTortho.subTensor(k);
          const int n = X.n();

          // XR = X_k * R_{k+1} (all mode indices); at the right boundary XR = X_k
          Tensor3<T> XR;
          if( k == nDim-1 )
            copy(X, XR);
          else
            normalize_contract2(X, R, XR);

          // maxvol on the adjoint of the right unfolding (n |J_{k+1}| x r_{k-1}) selects the columns J_k
          Tensor2<T> VtH(XR.n()*XR.r2(), XR.r1());
          EigenMap(VtH) = ConstEigenMap(unfold_right(XR)).adjoint();
          const std::vector<int> columns = maxvol(VtH, maxvolTolerance);
          J[k] = rightIndicesFromColumns(columns, n);

          // R_k = the selected columns (three-index access, the same as in interface_loop_from_right)
          R.resize(XR.r1(), J[k].size());
          for(int b = 0; b < J[k].size(); b++)
            for(int i = 0; i < XR.r1(); i++)
              R(i, b) = XR(i, J[k].idx[b], J[k].link[b]);
        }

        return J;
      }
    }
  }
}


#endif // PITTS_TENSORTRAIN_CROSS_HELPER_IMPL_HPP
