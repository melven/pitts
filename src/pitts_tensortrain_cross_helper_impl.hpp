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
#include <algorithm>
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
      // Possible deduplication: enrichSubspace in solveMALS could call this function (it differs in the Gram
      // correction, the drop tolerance and the second projection). Not done for now, because it would change
      // solveMALS, which is used for performance measurements and has been tested extensively in its current form.
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


      // implement solveLeftInterface
      template<typename T>
      void solveLeftInterface(const Tensor2<T>& L, const Tensor3<T>& B, Tensor3<T>& C)
      {
        const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

        if( L.r1() != B.r1() )
          throw std::invalid_argument("solveLeftInterface: dimension mismatch!");

        // least squares L C = B on the right unfolding of B (q x n m); L is tall but only slightly (q = r + oversampling),
        // so a pivoted QR and no TSQR
        const Eigen::ColPivHouseholderQR<Eigen::MatrixX<T>> qr(ConstEigenMap(L));
        C.resize(L.r2(), B.n(), B.r2());
        EigenMap(unfold_right(C)) = qr.solve(ConstEigenMap(unfold_right(B)));
      }


      // implement solveRightInterface
      template<typename T>
      void solveRightInterface(const Tensor3<T>& B, const Tensor2<T>& R, Tensor3<T>& C)
      {
        const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

        if( R.r2() != B.r2() )
          throw std::invalid_argument("solveRightInterface: dimension mismatch!");

        // mirror image of solveLeftInterface on the left unfolding of B (m n x q): C R = B <=> R^H C^H = B^H
        // (adjoint() is only a view; the solve copies its right-hand side into a block-sized temporary in both cases)
        const Eigen::ColPivHouseholderQR<Eigen::MatrixX<T>> qr(ConstEigenMap(R).adjoint());
        C.resize(B.r1(), B.n(), R.r1());
        EigenMap(unfold_left(C)) = qr.solve(ConstEigenMap(unfold_left(B)).adjoint()).adjoint();
      }


      // implement updateCoreLeft
      template<typename T>
      CoreUpdate<T> updateCoreLeft(const Tensor3<T>& C, const Tensor2<T>& leftInterface, T rankTolerance, int maxRank, T maxvolTolerance,
                                   Tensor3<T> enrichment, T oversampling, int maxEnrichment, int maxEnrichedRank, int maxRows)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int r1 = C.r1();
        const int n = C.n();
        const int r2 = C.r2();
        if( r1*n*r2 == 0 )
          throw std::invalid_argument("updateCoreLeft: empty block!");
        if( leftInterface.r2() != r1 )
          throw std::invalid_argument("updateCoreLeft: dimension mismatch of the left interface!");

        CoreUpdate<T> result;

        // 1. truncated SVD of the left unfolding: C ~ U * SVt with orthonormal columns of U
        //    (truncation in the Frobenius norm as in PITTS::normalize: the discarded part is bounded by
        //     rankTolerance * ||C||_F, whereas a threshold on the individual singular values would allow
        //     an error of sqrt(#discarded) * rankTolerance * sigma_1)
        auto [U, SVt] = normalize_svd(unfold_left(C), true, rankTolerance, maxRank, false, true);
        result.truncatedRank = U.r2();

        // 2. enrichment: append new orthonormal directions to U and zero rows to SVt
        //    (this does not change U*SVt, it only enlarges the space the next core can use;
        //     the projection works on the core-shaped factors, folding and unfolding are moves)
        Tensor3<T> U3 = fold_left(std::move(U), n);
        if( enrichment.r2() > 0 )
        {
          // directions below the truncation tolerance of the block are irrelevant (rankTolerance 0: only drop noise)
          using RealType = decltype(std::abs(T()));
          const RealType sqrtEps = std::sqrt(std::numeric_limits<RealType>::epsilon());
          const RealType relTol = std::real(rankTolerance) > 0 ? std::min(sqrtEps, RealType(std::real(rankTolerance))) : sqrtEps;
          enrichFactors(U3, SVt, std::move(enrichment), std::min<long long>(maxEnrichment, maxEnrichedRank - U3.r2()), T(relTol));
        }
        const int r = U3.r2();

        // 3. maxvol on the values of the new left part of y on I_{k-1} x S_k selects the rows I_k
        //    (with oversampling more than r)
        Tensor3<T> M;
        normalize_contract1(leftInterface, U3, M);
        const int nRows = std::max(r, std::min(r + static_cast<int>(std::ceil(std::real(oversampling) * r)), maxRows));
        result.selected = maxvol<T>(unfold_left(M), maxvolTolerance, 100, nullptr, nRows);

        result.core = std::move(U3);
        result.carry = std::move(SVt);
        return result;
      }


      // implement updateCoreRight
      template<typename T>
      CoreUpdate<T> updateCoreRight(const Tensor3<T>& C, const Tensor2<T>& rightInterface, T rankTolerance, int maxRank,
                                    T maxvolTolerance, T oversampling, int maxColumns)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int r1 = C.r1();
        const int n = C.n();
        const int r2 = C.r2();
        if( r1*n*r2 == 0 )
          throw std::invalid_argument("updateCoreRight: empty block!");
        if( rightInterface.r1() != r2 )
          throw std::invalid_argument("updateCoreRight: dimension mismatch of the right interface!");

        CoreUpdate<T> result;

        // 1. truncated SVD of the right unfolding: C ~ US * Vt with orthonormal rows of Vt
        //    (truncation in the Frobenius norm, see updateCoreLeft)
        auto [US, Vt] = normalize_svd(unfold_right(C), false, rankTolerance, maxRank, false, true);
        const int r = Vt.r1();
        result.truncatedRank = r;
        Tensor3<T> Vt3 = fold_right(std::move(Vt), n);

        // 2. maxvol on the values of the new right part of y on S_k x J_{k+1} selects the columns J_k
        //    (with oversampling more than r)
        Tensor3<T> M;
        normalize_contract2(Vt3, rightInterface, M);
        const int nColumns = std::max(r, std::min(r + static_cast<int>(std::ceil(std::real(oversampling) * r)), maxColumns));
        result.selected = maxvolColumns<T>(unfold_right(M), maxvolTolerance, nullptr, nColumns);

        result.core = std::move(Vt3);
        result.carry = std::move(US);
        return result;
      }


      // implement dominantRows
      template<typename T>
      std::vector<int> dominantRows(const Tensor3<T>& B, T absoluteTolerance, int nRows, T maxvolTolerance)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const Tensor2<T> U = normalize_svd(unfold_left(B), true, absoluteTolerance, nRows, true, true).first;
        return maxvol<T>(U, maxvolTolerance, 100, nullptr, nRows);
      }


      // implement dominantColumns
      template<typename T>
      std::vector<int> dominantColumns(const Tensor3<T>& B, T absoluteTolerance, int nColumns, T maxvolTolerance)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const Tensor2<T> Vt = normalize_svd(unfold_right(B), false, absoluteTolerance, nColumns, true, true).second;
        return maxvolColumns<T>(Vt, maxvolTolerance, nullptr, nColumns);
      }


      // implement initialRightFrame
      template<typename T>
      void initialRightFrame(const std::vector<std::reference_wrapper<const TensorTrain<T>>>& TTx, TensorTrain<T>& TTy, int maxRank)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const std::vector<int>& dims = TTy.dimensions();
        const int nDim = dims.size();
        if( TTx.empty() )
          throw std::invalid_argument("initialRightFrame: no tensor train!");
        for(const TensorTrain<T>& TT: TTx)
          if( TT.dimensions() != dims || TT.subTensor(0).r1() != 1 || TT.subTensor(nDim-1).r2() != 1 )
            throw std::invalid_argument("initialRightFrame: dimensions or boundary ranks of the inputs do not match!");

        // each input separately: minimal ranks (a right sweep alone does not reveal deficiencies in the left unfoldings)
        // and unit norm, with left-orthonormal cores. So the right parts of all inputs have the singular values of their own
        // unfoldings, independent of their norms and of the gauge of their cores (in a direct sum of the unnormalized inputs,
        // the truncation of round-off below would drop all directions of an input that is small compared to another input).
        std::vector<TensorTrain<T>> TTortho;
        TTortho.reserve(TTx.size());
        for(const TensorTrain<T>& TT: TTx)
        {
          copy(TT, TTortho.emplace_back(dims));
          leftNormalize(TTortho.back(), T(0));
        }

        // direct sum of the inputs: block-diagonal cores (the first core concatenated along r2, the last along r1),
        // so the right parts span the right parts of all inputs
        std::vector<Tensor3<T>> cores(nDim);
        for(int k = 0; k < nDim; k++)
        {
          long long r1 = 0, r2 = 0;
          for(const TensorTrain<T>& TT: TTortho)
          {
            r1 += TT.subTensor(k).r1();
            r2 += TT.subTensor(k).r2();
          }
          if( k == 0 )
            r1 = 1;
          if( k == nDim-1 )
            r2 = 1;
          cores[k].resize(r1, dims[k], r2);
          cores[k].setConstant(T(0));
          long long offset1 = 0, offset2 = 0;
          for(const TensorTrain<T>& TT: TTortho)
          {
            const auto& X = TT.subTensor(k);
            for(int c = 0; c < X.r2(); c++)
              for(int j = 0; j < X.n(); j++)
                for(int a = 0; a < X.r1(); a++)
                  cores[k](offset1+a, j, offset2+c) = X(a, j, c);
            if( k > 0 )
              offset1 += X.r1();
            if( k < nDim-1 )
              offset2 += X.r2();
          }
        }
        TensorTrain<T> frame(std::move(cores));

        // right-orthonormal cores of the direct sum: the right unfoldings stack the right parts of all inputs, so tolerance 0
        // only removes round-off and (numerically) dependent directions of different inputs. Otherwise only maxRank and the
        // dimensions on the left truncate (the summed ranks may exceed n_0 * ... * n_{k-1} at the bond between cores k-1 and k).
        // No left sweep over the direct sum: it would mix the inputs, e.g. directions that cancel in x1 + x2 would be lost.
        std::vector<int> leftLimit(nDim);
        long long nLeft = 1;
        for(int k = 1; k < nDim; k++)
        {
          nLeft = std::min<long long>(nLeft * dims[k-1], std::numeric_limits<int>::max());
          leftLimit[k] = std::min<long long>(nLeft, maxRank);
        }
        for(int k = nDim-1; k > 0; k--)
          rightNormalize_range(frame, k-1, k, T(0), leftLimit[k]);
        TTy.setSubTensors(0, std::move(frame));

        Tensor3<T> zero(1, dims[0], TTy.subTensor(0).r2());
        zero.setConstant(T(0));
        TTy.setSubTensor(0, std::move(zero));
      }


      // implement rightIndexSetsFromFrame
      template<typename T>
      std::vector<NestedIndices> rightIndexSetsFromFrame(const TensorTrain<T>& TT, T maxvolTolerance, T oversampling,
                                                         const std::vector<int>& maxIndices)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const std::vector<int>& dims = TT.dimensions();
        const int nDim = dims.size();
        if( !maxIndices.empty() && std::ssize(maxIndices) != nDim-1 )
          throw std::invalid_argument("rightIndexSetsFromFrame: maxIndices needs one entry per bond!");

        // right-to-left sweep of the partial contractions XR_k = X_k R_{k+1}; J[k+1] is set before XR_k is computed,
        // so the same recursion (and boundary handling) as in the cross sweeps can be used
        std::vector<NestedIndices> J(nDim);
        const auto neverFromTheLeft = [](int, optional_cref<Tensor3<T>>, Tensor3<T>&) { assert(false); };
        auto Rx = defineSweepData<Tensor3<T>>(nDim, neverFromTheLeft, contraction_loop_from_right<T>(TT, J));
        for(int k = nDim-1; k >= 1; k--)
        {
          Rx.update(-1, k);
          const Tensor3<T>& XR = Rx.right()->get();
          const int r = XR.r1();
          const int maxColumns = maxIndices.empty() ? std::numeric_limits<int>::max() : maxIndices[k-1];
          const int nColumns = std::max(r, std::min(r + static_cast<int>(std::ceil(std::real(oversampling) * r)), maxColumns));
          J[k] = rightIndicesFromColumns(maxvolColumns<T>(unfold_right(XR), maxvolTolerance, nullptr, nColumns), dims[k]);
        }
        return J;
      }
    }
  }
}


#endif // PITTS_TENSORTRAIN_CROSS_HELPER_IMPL_HPP
