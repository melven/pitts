// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_cross_helper.hpp
* @brief Building blocks for TT-cross algorithms: nested index sets, interface matrices, core updates
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-11
*
* Notation and references (see the file header of pitts_tensortrain_cross.hpp): a tensor train with d cores X_k of shape r_{k-1} x n_k x r_k,
* nested left index sets I_k (bond k, k = -1..d-2, I_{-1} = {()}) and nested right index sets J_k
* (bond k-1, k = 1..d, J_d = {()}).
*
* Index set numbering: I_k contains multi-indices of the dimensions 0..k, J_k contains multi-indices of
* the dimensions k..d-1. So the two sets of the same bond k are I_k and J_{k+1}, and a local problem on
* the cores k..k+m (SweepIndex with leftDim = k, rightDim = k+m, m = 0 for ALS-cross, m = 1 for DMRG-cross)
* is the block x(I_{k-1}, S_k x ... x S_{k+m}, J_{k+m+1}) = L_{k-1} * X_k * ... * X_{k+m} * R_{k+m+1};
* the sets I_k..I_{k+m-1} and J_{k+1}..J_{k+m} inside the block are not used and are recomputed by the
* update. In vectors, I[k] and J[k] store I_k and J_k (J[0] is unused, J_d and I_{-1} are implicit).
*
* Index conventions follow the unfoldings in pitts_tensor3_unfold.hpp:
*   left unfolding  (r_{k-1} n_k) x r_k : row    = a + i*r_{k-1}   for (a in I_{k-1}, i in S_k)
*   right unfolding r_{k-1} x (n_k r_k) : column = i + b*n_k       for (i in S_k, b in J_{k+1})
*
* Sketch of one left-to-right step at core k with these helpers (for a TT x and its cross sets I, J;
* Lx is a SweepData of the partial contractions, see contraction_loop_from_left):
*
*   Lx.update(k, k+1);                              // LX_k = L_{k-1} X_k and XR_{k+1} = X_{k+1} R_{k+2} valid
*   localBlockFromLeft(*Lx.left(), x, k, 1, Lx.right(), J, B);   // B = LX_k * R_{k+1}  (|I_{k-1}| x n_k x |J_{k+1}|)
*   applyToBlock(f, B);                             // B  = f(x) on the cross
*   // y in the orthogonal gauge with the interface matrices L^y_{k-1} = interfaceLeft(LY_{k-1}, I_{k-1}) and
*   // R^y_{k+1} = interfaceRight(YR_{k+1}, J_{k+1}) from its partial contractions (1x1 identities at the boundary):
*   solveLeftInterface(L^y_{k-1}, B, BL);           // core coordinates: C = L^y_{k-1}^+ B R^y_{k+1}^+
*   solveRightInterface(BL, R^y_{k+1}, C);
*   auto upd = updateCoreLeft(C, L^y_{k-1}, tol, maxRank, maxvolTol, enrichment);
*   I[k] = leftIndicesFromRows(upd.selected, |I_{k-1}|);
*   Lx.invalidate(k+1);                             // LX_{k+1} depends on the new I_k, recomputed on the next update
*   Y_k = upd.core (left-orthonormal);  Y_{k+1} = upd.carry * Y_{k+1};
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_CROSS_HELPER_HPP
#define PITTS_TENSORTRAIN_CROSS_HELPER_HPP

// includes
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <concepts>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include "pitts_multivector.hpp"
#include "pitts_random.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensor3_combine.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby_normalized.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_sweep_data.hpp"
#include "pitts_timer.hpp"
#include "pitts_eigen.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! dedicated helper functionality for TT-cross algorithms
    namespace tt_cross
    {
      //! Nested index set of one bond, stored by reference to the neighboring set
      //!
      //! Left sets I_k: element a is the multi-index (I_{k-1}[link[a]], idx[a]),
      //! right sets J_k: element b is the multi-index (idx[b], J_{k+1}[link[b]]).
      //!
      //! Why not store full multi-indices: the cross algorithms only ever need (a) the position of the
      //! parent multi-index to pick the matching row of the neighboring interface matrix and (b) the
      //! own mode index to pick the slice of the core. Both are exactly what a row of the left unfolding
      //! (r_{k-1} n_k x r_k) encodes, so a maxvol row selection converts to a nested set in O(1) per
      //! element and nestedness holds by construction. Full multi-indices are only recovered for
      //! tests, diagnostics and the final TTCrossInfo (expandLeft / expandRight).
      //!
      //! Note: a left set is only valid relative to the I_{k-1} it was created from. When the sweep
      //! replaces I_{k-1}, the old I_k is not nested any more (it still refers to the old parent
      //! positions) and must be recomputed before it is used; this is inherent to cross sweeps
      //! (Dolgov/Savostyanov 2020, Remark 5, reference [3] in pitts_tensortrain_cross.hpp) and handled by
      //! the sweep order in the driver.
      //!
      struct NestedIndices
      {
        //! empty set
        NestedIndices() = default;

        //! set with n (uninitialized) multi-indices
        explicit NestedIndices(int n) : link(n), idx(n) {}

        //! position in the neighboring (parent) index set: I_{k-1} for left sets, J_{k+1} for right sets
        std::vector<int> link;

        //! mode index i_k of the own dimension
        std::vector<int> idx;

        //! number of multi-indices in the set (= TT rank of the bond)
        [[nodiscard]] int size() const {return static_cast<int>(idx.size());}

        //! same link and idx entries (the same multi-indices in the same order if the neighboring sets are the same)
        bool operator==(const NestedIndices&) const = default;
      };

      //! decode a row selection of a left unfolding (r_prev*n x r) into a nested left index set
      //!
      //! This is the inverse of the row numbering of unfold_left (row = i1 + j*r1) and the only place
      //! where this index arithmetic appears: maxvol works on the unfolded matrix and returns row
      //! numbers, everything else accesses cores with three indices (link, idx, :) again.
      //!
      //! @param rows     selected rows (e.g. from maxvol), each row = link + idx*r_prev
      //! @param r_prev   number of rows per mode index (= |I_{k-1}|)
      //!
      inline NestedIndices leftIndicesFromRows(const std::vector<int>& rows, int r_prev)
      {
        if( r_prev < 1 )
          throw std::invalid_argument("leftIndicesFromRows: r_prev must be positive!");
        NestedIndices I(rows.size());
        for(int a = 0; a < rows.size(); a++)
        {
          if( rows[a] < 0 )
            throw std::invalid_argument("leftIndicesFromRows: negative row index!");
          I.link[a] = rows[a] % r_prev;
          I.idx[a] = rows[a] / r_prev;
        }
        return I;
      }

      //! decode a column selection of a right unfolding (r x n*r_next) into a nested right index set
      //!
      //! Inverse of the column numbering of unfold_right (column = j + k*n), see leftIndicesFromRows.
      //!
      //! @param columns  selected columns (e.g. from maxvol of the transposed unfolding), each column = idx + link*n
      //! @param n        mode size
      //!
      inline NestedIndices rightIndicesFromColumns(const std::vector<int>& columns, int n)
      {
        if( n < 1 )
          throw std::invalid_argument("rightIndicesFromColumns: n must be positive!");
        NestedIndices J(columns.size());
        for(int b = 0; b < columns.size(); b++)
        {
          if( columns[b] < 0 )
            throw std::invalid_argument("rightIndicesFromColumns: negative column index!");
          J.idx[b] = columns[b] % n;
          J.link[b] = columns[b] / n;
        }
        return J;
      }

      //! Mirror the left index sets of a symmetric tensor to the right: J_{d-1-k} = I_k reversed, for k = 0..nLeft-1
      //!
      //! For a tensor with y(i_0, ..., i_{d-1}) = y(i_{d-1}, ..., i_0) (and n_k = n_{d-1-k}), the right multi-index
      //! (i_{d-1-k}, ..., i_{d-1}) of J_{d-1-k} is the reversed left multi-index (i_0, ..., i_k) of I_k. In the nested
      //! representation both are the same data: the own mode index (of dimension k resp. d-1-k) and the position in the
      //! neighboring set (I_{k-1} resp. its mirror J_{d-k}), so the sets are just copied.
      //!
      //! @param I      left index sets, I[0..nLeft-1] are mirrored
      //! @param nLeft  number of mirrored sets (at most d-1)
      //! @param J      right index sets of size d (J[0] unused): J[d-1-k] is overwritten for k = 0..nLeft-1
      //!
      inline void mirrorIndexSets(const std::vector<NestedIndices>& I, int nLeft, std::vector<NestedIndices>& J)
      {
        const int nDim = J.size();
        if( nLeft < 0 || nLeft > static_cast<int>(I.size()) || nLeft > nDim-1 )
          throw std::invalid_argument("mirrorIndexSets: invalid number of sets!");
        for(int k = 0; k < nLeft; k++)
          J[nDim-1-k] = I[k];
      }

      //! random right-nested index sets J[k] (k = 1..d-1) with at most `size` elements each, drawn without replacement
      //!
      //! Used for the initial error crosses of the AMEn-style enrichment (as in amen_cross); the sets are
      //! nested by construction (each J_k is drawn from S_k x J_{k+1}). Uses internal::randomGenerator.
      //!
      inline std::vector<NestedIndices> randomRightIndexSets(const std::vector<int>& dims, int size)
      {
        if( size < 1 )
          throw std::invalid_argument("randomRightIndexSets: size must be positive!");
        const int nDim = dims.size();
        std::vector<NestedIndices> J(nDim);
        for(int k = nDim-1; k >= 1; k--)
        {
          // all (idx, link) candidates of S_k x J_{k+1} in a shuffled order, take the first ones
          const int nCandidates = dims[k] * ((k == nDim-1) ? 1 : J[k+1].size());
          std::vector<int> candidates(nCandidates);
          std::iota(candidates.begin(), candidates.end(), 0);
          std::shuffle(candidates.begin(), candidates.end(), randomGenerator);
          candidates.resize(std::min(size, nCandidates));
          J[k] = rightIndicesFromColumns(candidates, dims[k]);
        }
        return J;
      }

      //! left interface matrix L_k (|I_k| x r_k) as the rows (link, idx) of the contraction LX_k = L_{k-1} X_k (|I_{k-1}| x n_k x r_k)
      //!
      //! L_k(a,:) = LX_k(link[a], idx[a], :), see contraction_loop_from_left.
      //!
      template<typename T>
      Tensor2<T> interfaceLeft(const Tensor3<T>& LX, const NestedIndices& I)
      {
        const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

        Tensor2<T> L(I.size(), LX.r2());
        for(int a = 0; a < I.size(); a++)
        {
          assert(I.link[a] >= 0 && I.link[a] < LX.r1() && I.idx[a] >= 0 && I.idx[a] < LX.n());
          for(int j = 0; j < LX.r2(); j++)
            L(a, j) = LX(I.link[a], I.idx[a], j);
        }
        return L;
      }

      //! right interface matrix R_k (r_{k-1} x |J_k|) as the columns (idx, link) of the contraction XR_k = X_k R_{k+1} (r_{k-1} x n_k x |J_{k+1}|)
      //!
      //! R_k(:,b) = XR_k(:, idx[b], link[b]), see contraction_loop_from_right.
      //!
      template<typename T>
      Tensor2<T> interfaceRight(const Tensor3<T>& XR, const NestedIndices& J)
      {
        const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

        Tensor2<T> R(XR.r1(), J.size());
        for(int b = 0; b < J.size(); b++)
        {
          assert(J.link[b] >= 0 && J.link[b] < XR.r2() && J.idx[b] >= 0 && J.idx[b] < XR.n());
          for(int i = 0; i < XR.r1(); i++)
            R(i, b) = XR(i, J.idx[b], J.link[b]);
        }
        return R;
      }

      //! C = L_k * B with the left interface L_k of the cached contraction LX (nullopt at the left boundary: C = B)
      //!
      //! The boundary interface L_{-1} is the 1x1 identity, so nothing has to be contracted there; this is the same
      //! std::nullopt convention as for the partial contractions of solveMALS (see SweepData::left()).
      //!
      //! @param LX    cached contraction LX_k of the neighboring dimension, nullopt at the left boundary
      //! @param I     the left index sets
      //! @param k     dimension of LX: I[k] (the set of bond k) selects its rows; pass iDim-1 for a block
      //!              starting at core iDim (unused at the boundary)
      //! @param B     block to contract (r_k x n x m)
      //! @param C     result (|I_k| x n x m)
      //!
      template<typename T>
      void contractLeftInterface(optional_cref<Tensor3<T>> LX, const std::vector<NestedIndices>& I, int k,
                                 const Tensor3<T>& B, Tensor3<T>& C)
      {
        if( LX )
          normalize_contract1(interfaceLeft(LX->get(), I.at(k)), B, C);
        else
          copy(B, C);
      }

      //! C = B * R_k with the right interface R_k of the cached contraction XR (nullopt at the right boundary: C = B)
      //!
      //! At the right boundary, the interface is the 1x1 identity, or the given boundary matrix if the tensor train
      //! continues beyond its last core: e.g. a symmetric tensor H H^T given by its left half H, whose right part at
      //! the middle bond is known from the left part (see FunctionOnCrosses::mirrorRightBoundary).
      //!
      //! @param B          block to contract (m x n x r_{k-1})
      //! @param XR         cached contraction XR_k of the neighboring dimension, nullopt at the right boundary
      //! @param J          the right index sets
      //! @param k          dimension of XR: J[k] (the set of bond k-1) selects its columns; pass iDim+1 for a
      //!                   block ending at core iDim (unused at the boundary)
      //! @param C          result (m x n x |J_k|)
      //! @param boundary   interface matrix at the right boundary (r_{k-1} x q), nullopt for the 1x1 identity
      //!
      template<typename T>
      void contractRightInterface(const Tensor3<T>& B, optional_cref<Tensor3<T>> XR, const std::vector<NestedIndices>& J, int k,
                                  Tensor3<T>& C, optional_cref<Tensor2<T>> boundary = std::nullopt)
      {
        if( XR )
          normalize_contract2(B, interfaceRight(XR->get(), J.at(k)), C);
        else if( boundary )
          normalize_contract2(B, boundary->get(), C);
        else
          copy(B, C);
      }

      //! C = L^+ * B: least-squares solution of L * C = B with a left interface matrix L (inverse of normalize_contract1)
      //!
      //! Converts a block sampled on the rows of a left index set into the coordinates of the TT (y in the orthogonal
      //! gauge, see pitts_tensortrain_cross.hpp). L (q x r, q >= r, the 1x1 identity at the left boundary) must have
      //! full column rank, which the maxvol selection of the index set guarantees; with q > r (oversampling) this is
      //! the least-squares fit on the q rows. Solved with a Householder QR of L without pivoting (backward stable for full
      //! rank, no inverse is formed).
      //!
      //! @param L     left interface matrix (q x r)
      //! @param B     block (q x n x m)
      //! @param C     result (r x n x m), may not alias B
      //!
      template<typename T>
      void solveLeftInterface(const Tensor2<T>& L, const Tensor3<T>& B, Tensor3<T>& C);

      //! C = B * R^+: least-squares solution of C * R = B with a right interface matrix R (inverse of normalize_contract2)
      //!
      //! Mirror image of solveLeftInterface: R (r x q, full row rank, the 1x1 identity at the right boundary),
      //! solved as R^H C^H = B^H with a Householder QR of R^H.
      //!
      //! @param B     block (m x n x q)
      //! @param R     right interface matrix (r x q)
      //! @param C     result (m x n x r), may not alias B
      //!
      template<typename T>
      void solveRightInterface(const Tensor3<T>& B, const Tensor2<T>& R, Tensor3<T>& C);

      //! the 1x1 identity: the interface matrix at the boundary of a tensor train
      template<typename T>
      Tensor2<T> boundaryInterface();

      //! B = L * X * R: a block in the coordinates of a TT (its core(s) X) onto the cross of the interface matrices L and R
      //!
      //! @param L     left interface matrix (q1 x r1), boundaryInterface() at the left boundary
      //! @param X     core(s) (r1 x n x r2)
      //! @param R     right interface matrix (r2 x q2), boundaryInterface() at the right boundary
      //! @param B     result (q1 x n x q2)
      //!
      template<typename T>
      void contractInterfaces(const Tensor2<T>& L, const Tensor3<T>& X, const Tensor2<T>& R, Tensor3<T>& B);

      //! C = L^+ * B * R^+: a block sampled on a cross into the coordinates of the TT (inverse of contractInterfaces)
      //!
      //! Least-squares solves with the interface matrices, see solveLeftInterface and solveRightInterface.
      //!
      //! @param L     left interface matrix (q1 x r1, full column rank)
      //! @param B     block (q1 x n x q2)
      //! @param R     right interface matrix (r2 x q2, full row rank)
      //! @param C     result (r1 x n x r2), may not alias B
      //!
      template<typename T>
      void solveInterfaces(const Tensor2<T>& L, const Tensor3<T>& B, const Tensor2<T>& R, Tensor3<T>& C);

      //! full multi-indices of the left set I_k as a matrix with one row (i_0,...,i_k) per element (I[0] is the set of bond 0)
      //!
      //! expandLeft(I, -1) returns the single empty multi-index of I_{-1} = {()} (a 1 x 0 matrix).
      //!
      inline Eigen::MatrixXi expandLeft(const std::vector<NestedIndices>& I, int k)
      {
        // follow the links from bond k back to bond 0, filling one column per step
        Eigen::MatrixXi result(k < 0 ? 1 : I.at(k).size(), k+1);
        for(int a = 0; a < result.rows(); a++)
        {
          int pos = a;
          for(int j = k; j >= 0; j--)
          {
            result(a, j) = I[j].idx.at(pos);
            pos = I[j].link[pos];
          }
        }
        return result;
      }

      //! full multi-indices of the right set J_k as a matrix with one row (i_k,...,i_{d-1}) per element (J[k] is the set of bond k-1, J[0] unused)
      //!
      //! expandRight(J, nDim, nDim) returns the single empty multi-index of J_d = {()} (a 1 x 0 matrix).
      //!
      inline Eigen::MatrixXi expandRight(const std::vector<NestedIndices>& J, int k, int nDim)
      {
        // follow the links from bond k-1 forward to the last dimension, filling one column per step
        Eigen::MatrixXi result(k >= nDim ? 1 : J.at(k).size(), nDim-k);
        for(int b = 0; b < result.rows(); b++)
        {
          int pos = b;
          for(int j = k; j < nDim; j++)
          {
            result(b, j-k) = J[j].idx.at(pos);
            pos = J[j].link[pos];
          }
        }
        return result;
      }


      //! Partial contractions of a TT on nested index sets, cached with SweepData
      //!
      //! Use with defineSweepData<Tensor3<T>>(nDim, contraction_loop_from_left<T>(TT, I), contraction_loop_from_right<T>(TT, J)):
      //!   subTensor(k) from the left  = LX_k = L_{k-1} X_k of shape |I_{k-1}| x n_k x r_k  (all mode indices of core k),
      //!   subTensor(k) from the right = XR_k = X_k R_{k+1} of shape r_{k-1} x n_k x |J_{k+1}|,
      //! where the interface matrices L_k (|I_k| x r_k, rows X_0(i_0) ... X_k(i_k) for (i_0..i_k) in I_k) and
      //! R_k (r_{k-1} x |J_k|, columns X_k(i_k) ... X_{d-1}(i_{d-1}) for (i_k..i_{d-1}) in J_k) are row/column
      //! selections of these: L_k = interfaceLeft(LX_k, I_k), R_k = interfaceRight(XR_k, J_k).
      //! The boundary interfaces L_{-1} and R_d are the 1x1 identity: SweepData reports them as std::nullopt
      //! (left()/right()) and the contraction is simply omitted (LX_0 = X_0, XR_{d-1} = X_{d-1}), as in solveMALS.
      //!
      //! The recursion L_k(a,:) = L_{k-1}(link[a],:) * X_k(:,idx[a],:) is why the index sets are stored as (link, idx).
      //! The cached quantity is the full contraction and not the interface, because the cross algorithms need it
      //! anyway: the block of core k on the cross (I_{k-1}, S_k, J_{k+1}) is LX_k R_{k+1}, the enrichment residual
      //! is LX_k R^z_{k+1}, and the next interface L_k is a selection of rows of the same LX_k. So each contraction
      //! is done once per core and sweep direction (this is what defineSweepData is for).
      //!
      //! Dependencies: LX_k depends on LX_{k-1}, I_{k-1} and X_k (not on I_k); XR_k on XR_{k+1}, J_{k+1} and X_k.
      //! So after I_k changed, invalidate(k+1); after J_k changed, invalidate(k-1); after X_k changed, invalidate(k).
      //! (invalidate drops the results from both sides at the given dimension, which costs nothing in a sweep:
      //! the other side is recomputed in the next step anyway.)
      //! Caveat: SweepData stores the left and right results in one array, so the valid ranges must not overlap
      //! (update(k, k+1) for the block of core k; update asserts newLeftDim < newRightDim).
      //! Used for x on the index sets of y, and for x and y on the index sets of the error approximation z.
      //!
      //! @param TT   the tensor train (captured by reference, must outlive the SweepData)
      //! @param I    the left index sets (captured by reference, I[k-1] must be valid when LX_k is computed)
      //! @return     lambda (int iDim, optional_cref<Tensor3<T>> prev, Tensor3<T>& result) for defineSweepData
      //!
      //! Defined in the header (like the *_loop_from_left helpers of solveMALS) because the deduced lambda
      //! type cannot be separated into an explicit instantiation; the actual work is in normalize_contract1.
      //!
      template<typename T>
      auto contraction_loop_from_left(const TensorTrain<T>& TT, const std::vector<NestedIndices>& I)
      {
        return [&](int iDim, optional_cref<Tensor3<T>> prev, Tensor3<T>& LX)
        {
          // LX_iDim = L_{iDim-1} X_iDim with L_{iDim-1} selected from the previous contraction
          contractLeftInterface<T>(prev, I, iDim-1, TT.subTensor(iDim), LX);
        };
      }

      //! right partial contractions for SweepData, see contraction_loop_from_left
      //!
      //! @param TT         the tensor train (captured by reference)
      //! @param J          the right index sets (captured by reference, J[k+1] must be valid when XR_k is computed)
      //! @param boundary   interface matrix at the right boundary, see contractRightInterface (captured by reference:
      //!                   its value may change, then the contractions must be invalidated)
      //!
      template<typename T>
      auto contraction_loop_from_right(const TensorTrain<T>& TT, const std::vector<NestedIndices>& J, optional_cref<Tensor2<T>> boundary = std::nullopt)
      {
        return [&TT, &J, boundary](int iDim, optional_cref<Tensor3<T>> prev, Tensor3<T>& XR)
        {
          // XR_iDim = X_iDim R_{iDim+1} with R_{iDim+1} selected from the previous contraction (or the boundary matrix)
          contractRightInterface<T>(TT.subTensor(iDim), prev, J, iDim+1, XR, boundary);
        };
      }

      //! type of the cached partial contractions of a TT on nested index sets (SweepData with the two loops above)
      template<typename T>
      using PartialContractions = decltype(defineSweepData<Tensor3<T>>(0,
          contraction_loop_from_left<T>(std::declval<const TensorTrain<T>&>(), std::declval<const std::vector<NestedIndices>&>()),
          contraction_loop_from_right<T>(std::declval<const TensorTrain<T>&>(), std::declval<const std::vector<NestedIndices>&>())));


      //! the cores iDim..iDim+nMALS-1 of a tensor train combined into a single block (modes combined as in combine)
      //!
      template<typename T>
      Tensor3<T> combineCores(const TensorTrain<T>& TT, int iDim, int nMALS)
      {
        Tensor3<T> block;
        copy(TT.subTensor(iDim), block);
        for(int i = iDim+1; i < iDim+nMALS; i++)
          block = combine(block, TT.subTensor(i));
        return block;
      }

      //! Local block of the cross for a left-to-right step: x(I_{iDim-1}, S_iDim x ... x S_{iDim+nMALS-1}, J_{iDim+nMALS})
      //!
      //! Counterpart of calculate_local_rhs in solveMALS: it assembles the cores of the local problem from the
      //! cached partial contractions of the sweep. The core at the side the sweep comes from is taken from the
      //! cache (LX = L_{iDim-1} X_iDim), the remaining cores of the block from TTx, and the interface of the
      //! other side from the neighboring cached contraction (nullopt at the boundary).
      //!
      //! @param LX      cached contraction LX_iDim (SweepData::left() after update(iDim, ...))
      //! @param TTx     tensor train, cores iDim+1 .. iDim+nMALS-1 are combined into the block
      //! @param iDim    first dimension of the block
      //! @param nMALS   number of dimensions of the block
      //! @param XR      cached contraction XR_{iDim+nMALS} (SweepData::right()), nullopt at the right boundary
      //! @param J         right index sets
      //! @param B         block (|I_{iDim-1}| x n_iDim...n_{iDim+nMALS-1} x |J_{iDim+nMALS}|, modes combined as in combine)
      //! @param boundary  interface matrix at the right boundary, see contractRightInterface
      //!
      template<typename T>
      void localBlockFromLeft(const Tensor3<T>& LX, const TensorTrain<T>& TTx, int iDim, int nMALS,
                              optional_cref<Tensor3<T>> XR, const std::vector<NestedIndices>& J, Tensor3<T>& B,
                              optional_cref<Tensor2<T>> boundary = std::nullopt)
      {
        if( nMALS == 1 )
          return contractRightInterface<T>(LX, XR, J, iDim+1, B, boundary);

        contractRightInterface<T>(combine(LX, combineCores(TTx, iDim+1, nMALS-1)), XR, J, iDim+nMALS, B, boundary);
      }

      //! Local block of the cross for a right-to-left step (mirror image of localBlockFromLeft)
      //!
      //! @param LX      cached contraction LX_{iDim-1} (SweepData::left()), nullopt at the left boundary
      //! @param I       left index sets
      //! @param TTx     tensor train, cores iDim .. iDim+nMALS-2 are combined into the block
      //! @param iDim    first dimension of the block
      //! @param nMALS   number of dimensions of the block
      //! @param XR      cached contraction XR_{iDim+nMALS-1} (SweepData::right() after update(..., iDim+nMALS-1))
      //! @param B       block, see localBlockFromLeft
      //!
      template<typename T>
      void localBlockFromRight(optional_cref<Tensor3<T>> LX, const std::vector<NestedIndices>& I,
                               const TensorTrain<T>& TTx, int iDim, int nMALS, const Tensor3<T>& XR, Tensor3<T>& B)
      {
        if( nMALS == 1 )
          return contractLeftInterface<T>(LX, I, iDim-1, XR, B);

        contractLeftInterface<T>(LX, I, iDim-1, combine(combineCores(TTx, iDim, nMALS-1), XR), B);
      }


      //! apply the element-wise function to every entry of a block (the only place where f is called)
      //!
      //! Two kinds of functors are supported, see applyElementwise:
      //!  * a scalar function T f(T): it must be pure (no shared state), the entries are processed in parallel;
      //!  * an array function f(MultiVector<T>&) transforming all entries of the block in-place. The block is
      //!    reshaped to a single column of r1*n*r2 rows without copying data, so the function is called once
      //!    per block (for functions with a per-call overhead, e.g. a numpy function called from Python) and
      //!    is itself responsible for any parallelization.
      //! A functor that accepts both signatures (e.g. a generic lambda) is used as an array function.
      //!
      //! For a function of several tensors, the blocks of the further arguments are passed after B (same
      //! shape, same entries of the respective tensors) and become additional arguments of f:
      //! T f(T, T, ...) resp. f(MultiVector<T>& B, const MultiVector<T>& other, ...) with the result in B.
      //!
      //! @tparam Function  scalar functor T f(T, ...) or array functor f(MultiVector<T>&, const MultiVector<T>&...)
      //! @param f          the element-wise function
      //! @param B          block of the first argument of f, overwritten with the result
      //! @param others     blocks of the further arguments of f (consumed: pass them with std::move)
      //! @return           number of function evaluations (= number of entries)
      //!
      //! @throws std::invalid_argument if the blocks have different shapes or an array function changes the dimensions of the block
      //!
      template<typename T, typename Function, typename... Others>
        requires (std::same_as<Others, Tensor3<T>> && ...)
      long long applyToBlock(const Function& f, Tensor3<T>& B, Others&&... others)
      {
        const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

        const auto r1 = B.r1();
        const auto n = B.n();
        const auto r2 = B.r2();
        if( ((others.r1() != r1 || others.n() != n || others.r2() != r2) || ...) )
          throw std::invalid_argument("TT-cross: the blocks of the function arguments have different shapes!");

        if constexpr( std::invocable<const Function&, MultiVector<T>&, std::conditional_t<true, const MultiVector<T>&, Others>...> )
        {
          MultiVector<T> entries = unfold(std::move(B));
          f(entries, unfold(std::move(others))...);
          if( entries.rows() != r1*n*r2 || entries.cols() != 1 )
            throw std::invalid_argument("TT-cross: the array function must not change the dimensions of the block!");
          B = fold(std::move(entries), r1, n, r2);
        }
        else
        {
          // simple entry-wise loop; the first index is contiguous in memory
#pragma omp parallel for collapse(2) schedule(static) if(r1*n*r2 > 500)
          for(long long k = 0; k < r2; k++)
            for(long long j = 0; j < n; j++)
              for(long long i = 0; i < r1; i++)
                B(i, j, k) = f(B(i, j, k), others(i, j, k)...);
        }

        return r1*n*r2;
      }


      //! uniformly distributed random multi-indices of the whole tensor, one per row
      //!
      //! In contrast to randomRightIndexSets these are independent: no nesting, so the entries sampled
      //! with them are uncorrelated (needed for an unbiased error estimate, see TTCrossOptions).
      //!
      inline Eigen::MatrixXi randomIndices(const std::vector<int>& dims, int nSamples)
      {
        Eigen::MatrixXi indices(nSamples, dims.size());
        for(int k = 0; k < int(dims.size()); k++)
        {
          std::uniform_int_distribution<int> distribution(0, dims[k]-1);
          for(int s = 0; s < nSamples; s++)
            indices(s, k) = distribution(randomGenerator);
        }
        return indices;
      }


      //! evaluate a tensor train at the given multi-indices (one per row), as a block of shape (nSamples,1,1)
      //!
      //! This is just a batched variant of the scalar products with the corresponding unit tensors
      //! (setUnit + dot), touching only the slice of each core that the index selects. The block shape
      //! allows reusing applyToBlock and rmsDifference on the result.
      //!
      //! With mirrored = true, TT is the left half H (m cores, open right bond) of the symmetric tensor x = H H^T
      //! with 2m dimensions: x(i_0, ..., i_{2m-1}) = <H(i_0, ..., i_{m-1}, :), H(i_{2m-1}, ..., i_m, :)>.
      //!
      //! @todo Move this to a dedicated top-level function (with a better implementation: group the
      //!       samples by index value into GEMMs, parallelize) as soon as it is needed somewhere else.
      //!
      template<typename T>
      void evaluateAt(const TensorTrain<T>& TT, const Eigen::MatrixXi& indices, Tensor3<T>& values, bool mirrored = false)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
        const int nSamples = indices.rows();
        const int nCores = TT.dimensions().size();

        // values of the cores 0..nCores-1 at the indices in the given columns: nSamples x r_nCores
        const auto partAt = [&](const auto& column)
        {
          EigenMatrix v = EigenMatrix::Ones(nSamples, 1);
          for(int k = 0; k < nCores; k++)
          {
            const auto& core = TT.subTensor(k);
            // slice core(:,j,:) as an r1 x r2 matrix (the Tensor3 memory is i1 + j*r1 + i2*r1*n)
            const auto slice = [&core](int j)
            {
              return Eigen::Map<const EigenMatrix, Eigen::Unaligned, Eigen::OuterStride<>>(
                  &core(0, j, 0), core.r1(), core.r2(), Eigen::OuterStride<>(core.r1()*core.n()));
            };

            EigenMatrix vNext(nSamples, core.r2());
            for(int s = 0; s < nSamples; s++)
              vNext.row(s) = v.row(s) * slice(indices(s, column(k)));
            v = std::move(vNext);
          }
          return v;
        };

        const EigenMatrix left = partAt([](int k){return k;});
        values.resize(nSamples, 1, 1);
        if( !mirrored )
        {
          for(int s = 0; s < nSamples; s++)
            values(s, 0, 0) = left(s, 0);
          return;
        }
        const EigenMatrix right = partAt([nCores](int k){return 2*nCores-1-k;});
        for(int s = 0; s < nSamples; s++)
          values(s, 0, 0) = left.row(s).cwiseProduct(right.row(s)).sum();   // transpose, not adjoint: x = H H^T
      }


      //! Element-wise function f(x^(1), ..., x^(N)) of N tensor trains, evaluated on the crosses of a TT-cross sweep
      //!
      //! Bundles everything of the TT-cross driver that depends on the input tensors: for each input the cached
      //! partial contractions (SweepData) on the index sets of y (I, J) and on those of the error approximation
      //! (Iz, Jz), the assembly of the block of each input on a cross, and the call of f (applyToBlock). So the
      //! driver itself (index sets, core updates, enrichment, convergence) is the same for any number of inputs.
      //!
      //! All block functions return f on the cross of core k (or of the cores k..k+nMALS-1) as a block of shape
      //! |left set| x n_k... x |right set|. They differ in the sets of the cross and in the side the cached
      //! contractions are taken from, which matters for the cost only (the same choices as in the single-input
      //! sketch at the top of this file). Each block function updates the contractions it needs itself.
      //!
      //! The index sets are owned by the caller and captured by reference: after changing a set, the caller must
      //! invalidate the affected contractions (invalidate / invalidateError, with the same rules as for
      //! contraction_loop_from_left: invalidate(k+1) after I_k changed, invalidate(k-1) after J_k changed).
      //!
      //! Mirrored inputs: each input is the left half H (m cores, open right bond) of a symmetric tensor x = H H^T with
      //! 2m dimensions, and the index sets are those of x (vectors of size 2m) with mirrored right sets at the middle
      //! bond (J_m = mirror(I_{m-1}) and J^z_m = mirror(I^z_{m-1}), see mirrorIndexSets). Only the cores of H are
      //! contracted: the right part of x at the middle bond is the transposed left part, so the right contractions
      //! end with a boundary matrix instead of the 1x1 identity (see mirrorRightBoundary). The blocks are then those of
      //! x on the crosses of the first m cores (and bondBlock(m) is the symmetric matrix at the middle bond).
      //!
      //! @tparam T         underlying data type
      //! @tparam Function  element-wise function with N arguments, see applyToBlock
      //! @tparam N         number of input tensor trains
      //!
      template<typename T, typename Function, std::size_t N>
      class FunctionOnCrosses final
      {
        public:
          //! the input tensor trains (all of the same dimensions)
          using Inputs = std::array<std::reference_wrapper<const TensorTrain<T>>, N>;

          //! setup without any calculation; f, the inputs and the index sets must outlive this object
          //!
          //! With mirroredInputs, the inputs are the left halves of symmetric tensors, see above (call
          //! mirrorRightBoundary before the first block).
          //!
          FunctionOnCrosses(const Function& f, const Inputs& TTx,
                            const std::vector<NestedIndices>& I, const std::vector<NestedIndices>& J,
                            const std::vector<NestedIndices>& Iz, const std::vector<NestedIndices>& Jz,
                            bool mirroredInputs = false) :
            f_(f), TTx_(TTx), I_(I), J_(J), Iz_(Iz), Jz_(Jz), mirrored_(mirroredInputs), Rx_(N), Rxz_(N)
          {
            for(int i = 0; i < int(N); i++)
            {
              const TensorTrain<T>& TT = TTx[i];
              const int nDim = TT.dimensions().size();
              Lx_.push_back(defineSweepData<Tensor3<T>>(nDim, contraction_loop_from_left<T>(TT, I), contraction_loop_from_right<T>(TT, J, boundary(Rx_, i))));
              Lxz_.push_back(defineSweepData<Tensor3<T>>(nDim, contraction_loop_from_left<T>(TT, Iz), contraction_loop_from_right<T>(TT, Jz, boundary(Rxz_, i))));
            }
          }

          //! no copies: the cached right contractions refer to the boundary matrices of this object (moving keeps them)
          FunctionOnCrosses(const FunctionOnCrosses&) = delete;
          FunctionOnCrosses& operator=(const FunctionOnCrosses&) = delete;
          FunctionOnCrosses(FunctionOnCrosses&&) = default;

          //! Right boundary matrices of mirrored inputs: the transposed left part of x at the middle bond
          //!
          //! For x = H H^T with H of m cores and the mirrored right sets J_m = mirror(I_{m-1}), the right part of x on J_m
          //! is R^x_m = (L^x_{m-1})^T with the values L^x_{m-1} of H on I_{m-1} (a row selection of the cached left
          //! contraction), and the same for the error sets. Call this after I_{m-1} or I^z_{m-1} changed (instead of
          //! invalidating): it recomputes the boundary matrices and invalidates all right contractions.
          //!
          void mirrorRightBoundary()
          {
            assert(mirrored_ && middleSetsMirrored());
            for(int i = 0; i < int(N); i++)
            {
              const int m = TTx_[i].get().dimensions().size();
              transposedLeftInterface(Lx_[i], I_, m, Rx_[i]);
              if( withErrorSets(m) )
                transposedLeftInterface(Lxz_[i], Iz_, m, Rxz_[i]);
            }
          }

          //! f on the cross (I_{k-1}, S_k x ... x S_{k+nMALS-1}, J_{k+nMALS}) with the cached LX_k (left-to-right step)
          Tensor3<T> blockFromLeft(int k, int nMALS)
          {
            return onCrosses([&](int i, Tensor3<T>& B)
            {
              Lx_[i].update(k, k+nMALS);
              localBlockFromLeft<T>(Lx_[i].left()->get(), TTx_[i].get(), k, nMALS, Lx_[i].right(), J_, B, boundary(Rx_, i));
            });
          }

          //! f on the cross (I_{kLeft-1}, S_kLeft x ... x S_{kLeft+nMALS-1}, J_{kLeft+nMALS}) with the cached XR (right-to-left step)
          Tensor3<T> blockFromRight(int kLeft, int nMALS)
          {
            return onCrosses([&](int i, Tensor3<T>& B)
            {
              Lx_[i].update(kLeft-1, kLeft+nMALS-1);
              localBlockFromRight<T>(Lx_[i].left(), I_, TTx_[i].get(), kLeft, nMALS, Lx_[i].right()->get(), B);
            });
          }

          //! f on the mixed cross (I_{k-1}, S_k, J^z_{k+1}): left sets of y, right sets of the error approximation (enrichment)
          Tensor3<T> mixedBlock(int k)
          {
            return onCrosses([&](int i, Tensor3<T>& B)
            {
              Lx_[i].update(k, k+1);
              Lxz_[i].update(k, k+1);
              contractRightInterface<T>(Lx_[i].left()->get(), Lxz_[i].right(), Jz_, k+1, B, boundary(Rxz_, i));
            });
          }

          //! f on the error cross (I^z_{k-1}, S_k, J^z_{k+1}) with the cached LX^z_k (left-to-right step)
          Tensor3<T> errorBlockFromLeft(int k)
          {
            return onCrosses([&](int i, Tensor3<T>& B)
            {
              Lxz_[i].update(k, k+1);
              contractRightInterface<T>(Lxz_[i].left()->get(), Lxz_[i].right(), Jz_, k+1, B, boundary(Rxz_, i));
            });
          }

          //! f on the error cross (I^z_{k-1}, S_k, J^z_{k+1}) with the cached XR^z_k (right-to-left step)
          Tensor3<T> errorBlockFromRight(int k)
          {
            return onCrosses([&](int i, Tensor3<T>& B)
            {
              Lxz_[i].update(k-1, k);
              contractLeftInterface<T>(Lxz_[i].left(), Iz_, k-1, Lxz_[i].right()->get(), B);
            });
          }

          //! f on the cross (I_{k-1}, J_k) of the bond between the cores k-1 and k, as a block |I_{k-1}| x 1 x |J_k|
          //!
          //! Without a mode index: the rows I_{k-1} and the columns J_k of the unfolding of f(x) at this bond (for a
          //! symmetric tensor with J_k the mirror of I_{k-1}, a symmetric matrix). Valid for 1 <= k <= number of cores
          //! (k = number of cores: the middle bond of mirrored inputs, with their boundary matrices).
          //!
          Tensor3<T> bondBlock(int k)
          {
            return onCrosses([&](int i, Tensor3<T>& B)
            {
              Lx_[i].update(k-1, k);
              contractRightInterface<T>(fold_left(interfaceLeft(Lx_[i].left()->get(), I_[k-1]), 1), Lx_[i].right(), J_, k, B, boundary(Rx_, i));
            });
          }

          //! f at the given multi-indices (one per row) as a block of shape (nSamples,1,1), see evaluateAt
          Tensor3<T> atIndices(const Eigen::MatrixXi& indices)
          {
            return evaluate([&](int i, Tensor3<T>& B){evaluateAt(TTx_[i].get(), indices, B, mirrored_);});
          }

          //! the contractions on the sets of y at dimension k are invalid (I or J changed)
          void invalidate(int k)
          {
            for(auto& L: Lx_)
              L.invalidate(k);
          }

          //! the contractions on the sets of the error approximation at dimension k are invalid (Iz or Jz changed)
          void invalidateError(int k)
          {
            for(auto& L: Lxz_)
              L.invalidate(k);
          }

          //! J_m (and J^z_m) is the mirror of I_{m-1} (and I^z_{m-1}), see mirrorIndexSets (checked in an assert, public for the tests)
          [[nodiscard]] bool middleSetsMirrored() const
          {
            const int m = TTx_[0].get().dimensions().size();
            return J_.at(m) == I_.at(m-1) && (!withErrorSets(m) || Jz_.at(m) == Iz_.at(m-1));
          }

          //! the boundary matrices fit to the inputs and the current sets I_{m-1} (and I^z_{m-1}), i.e. mirrorRightBoundary
          //! was called after the last change (checked in an assert, public for the tests; a changed set of the same size is not detected)
          [[nodiscard]] bool boundariesFit() const
          {
            for(int i = 0; i < int(N); i++)
            {
              const TensorTrain<T>& TT = TTx_[i];
              const int m = TT.dimensions().size();
              const int r = TT.subTensor(m-1).r2();
              if( Rx_[i].r1() != r || Rx_[i].r2() != I_[m-1].size() )
                return false;
              if( withErrorSets(m) && (Rxz_[i].r1() != r || Rxz_[i].r2() != Iz_[m-1].size()) )
                return false;
            }
            return true;
          }

          //! total number of evaluated entries (calls of a scalar f)
          [[nodiscard]] long long nEvaluations() const {return nEvaluations_;}

        private:
          //! the right boundary matrix of input i (nullopt = the 1x1 identity for inputs that are not mirrored)
          optional_cref<Tensor2<T>> boundary(const std::vector<Tensor2<T>>& R, int i) const
          {
            if( mirrored_ )
              return std::cref(R[i]);
            return std::nullopt;
          }

          //! the error sets of mirrored inputs are used (they may be empty without enrichment)
          bool withErrorSets(int m) const {return static_cast<int>(Iz_.size()) >= m && Iz_[m-1].size() > 0;}

          //! R = (L_{m-1})^T, the transposed rows I_{m-1} of the left contraction at the last core; invalidates the right contractions
          static void transposedLeftInterface(PartialContractions<T>& L, const std::vector<NestedIndices>& I, int m, Tensor2<T>& R)
          {
            L.update(m-1, m);
            const Tensor2<T> left = interfaceLeft(L.left()->get(), I[m-1]);
            R.resize(left.r2(), left.r1());
            EigenMap(R) = ConstEigenMap(left).transpose();   // transpose, not adjoint: x = H H^T (symmetric)
            L.invalidate(m-1);
          }

          //! evaluate on a cross, see evaluate (the right contractions of mirrored inputs end with the boundary matrices)
          template<typename BlockOfInput>
          Tensor3<T> onCrosses(const BlockOfInput& blockOfInput)
          {
            // mirrored inputs: mirrorRightBoundary must have been called after the last change of the middle sets
            assert(!mirrored_ || boundariesFit());
            return evaluate(blockOfInput);
          }

          //! assemble the block of every input with blockOfInput(i, B), then apply f to all of them
          template<typename BlockOfInput>
          Tensor3<T> evaluate(const BlockOfInput& blockOfInput)
          {
            std::array<Tensor3<T>, N> B;
            for(int i = 0; i < N; i++)
              blockOfInput(i, B[i]);
            nEvaluations_ += [&]<std::size_t... i>(std::index_sequence<i...>)
            {
              return applyToBlock(f_, B[0], std::move(B[i+1])...);
            }(std::make_index_sequence<N-1>{});
            return std::move(B[0]);
          }

          const Function& f_;
          Inputs TTx_;
          const std::vector<NestedIndices>& I_;
          const std::vector<NestedIndices>& J_;
          const std::vector<NestedIndices>& Iz_;
          const std::vector<NestedIndices>& Jz_;
          //! inputs are left halves of symmetric tensors
          const bool mirrored_;
          //! right boundary matrices of mirrored inputs on the sets of y and of the error approximation (fixed size N:
          //! the right contractions refer to the elements)
          std::vector<Tensor2<T>> Rx_;
          std::vector<Tensor2<T>> Rxz_;
          //! partial contractions of each input on the sets of y
          std::vector<PartialContractions<T>> Lx_;
          //! partial contractions of each input on the sets of the error approximation
          std::vector<PartialContractions<T>> Lxz_;
          long long nEvaluations_ = 0;
      };


      //! root-mean-square size of a block: ||A||_F / sqrt(number of entries)
      //!
      //! The blocks of a TT-cross sweep sample the tensor on crosses of different (and growing) size, so
      //! only this scaling makes their magnitudes comparable (a Frobenius norm grows with the TT rank).
      //!
      template<typename T>
      auto t3_rms(const Tensor3<T>& A)
      {
        using RealType = decltype(std::abs(T()));
        const auto nEntries = A.r1()*A.n()*A.r2();
        return nEntries > 0 ? t3_nrm(A) / std::sqrt(RealType(nEntries)) : RealType(0);
      }


      //! root-mean-square difference t3_rms(A - B) of two blocks of equal shape
      //!
      //! @throws std::invalid_argument for different shapes
      //!
      template<typename T>
      auto rmsDifference(const Tensor3<T>& A, const Tensor3<T>& B)
      {
        if( A.r1() != B.r1() || A.n() != B.n() || A.r2() != B.r2() )
          throw std::invalid_argument("rmsDifference: blocks have different shapes!");

        Tensor3<T> diff;
        copy(A, diff);
        t3_axpy(T(-1), B, diff);
        return t3_rms(diff);
      }


      //! Enrich the two factors of a truncated SVD by new directions of E, leaving their product unchanged
      //!
      //! Q is the core-shaped form (r1 x n x r) of a left-orthogonal matrix, B (r x l) the remainder, so
      //! unfold_left(Q)*B is the truncated block, and E (r1 x n x rho) holds candidate directions: the part
      //! of E outside the span of Q is orthonormalized and appended to Q, and B is padded with zero rows.
      //!
      //! Q is only orthogonal up to round-off (internal::normalize_svd reconstructs it as Q = M V S^{-1}),
      //! which the projection takes into account, see the implementation. Used by updateCoreLeft; declared
      //! here so that the projection can be tested with a Q of a prescribed orthogonality error.
      //! (This is the enrichSubspace step of solveMALS as a function.)
      //!
      //! @param Q              orthogonal factor as a core, extended in place
      //! @param B              remainder, padded with zero rows in place
      //! @param E              candidate directions as a core, consumed (the projection overwrites it)
      //! @param maxDirections  maximal number of appended directions; if E has more columns, the dominant
      //!                       directions of the projected E are chosen (SVD), as in solveMALS
      //! @param relativeTolerance  directions with a part outside span(Q) below relativeTolerance * ||E|| are dropped
      //!
      template<typename T>
      void enrichFactors(Tensor3<T>& Q, Tensor2<T>& B, Tensor3<T>&& E, int maxDirections = std::numeric_limits<int>::max(),
                         T relativeTolerance = T(std::sqrt(std::numeric_limits<decltype(std::abs(T()))>::epsilon())));


      //! Result of a local core update (updateCoreLeft / updateCoreRight)
      template<typename T>
      struct CoreUpdate
      {
        //! new core, orthonormal: left-orthonormal r1 x n x r' (left) or right-orthonormal r' x n x r2 (right)
        Tensor3<T> core;

        //! remainder to be multiplied into the neighboring core: r' x r2 (left, Y_{k+1} <- carry * Y_{k+1})
        //! or r1 x r' (right, Y_{k-1} <- Y_{k-1} * carry)
        Tensor2<T> carry;

        //! selected rows of (interface x I) * core (left) or columns of core * interface (right), size q >= r'
        std::vector<int> selected;

        //! rank r' after truncation (without enrichment) -- for convergence/rank reporting
        int truncatedRank = 0;
      };

      //! Local core update for a left-to-right sweep (y in the orthogonal gauge)
      //!
      //! Given the block C (r1 x n x r2) of f(x) on the cross (I_{k-1}, S_k, J_{k+1}) in the coordinates of y
      //! (converted with solveLeftInterface / solveRightInterface, so that L^y_{k-1} C R^y_{k+1} fits f(x) on the
      //! cross), with the cores left of k left-orthonormal and the cores right of k right-orthonormal:
      //!  1. truncated SVD of the left unfolding: C ~ U * SVt with orthonormal U (r1 n x r'), truncation in the
      //!     Frobenius norm with relative tolerance rankTolerance and maximal rank maxRank (internal::normalize_svd).
      //!     Because of the orthonormal frames this is the truncation of y itself, as in PITTS::normalize;
      //!  2. optional enrichment: the block E (r1 x n x rho, error directions in the same coordinates) is
      //!     orthogonalized against U and appended, U' = [U, Q_E], and SVt is padded with zero rows (see
      //!     enrichFactors, the enrichSubspace step of solveMALS);
      //!  3. maxvol on M = (L^y_{k-1} x I_n) U' (the values of the new left part of y on I_{k-1} x S_k) selects
      //!     the rows I_k, q = r' + ceil(oversampling r') of them (at most maxRows and the number of rows of M).
      //!
      //! The new core is U' (left-orthonormal) and the carry SVt', so core * carry = U' SVt' (= C up to the
      //! truncation error). The interface of the next step, L^y_k = M(rows,:), has full column rank.
      //!
      //! @param C                  block in the coordinates of y (r1 x n x r2)
      //! @param leftInterface      L^y_{k-1} (|I_{k-1}| x r1), the 1x1 identity at the left boundary
      //! @param rankTolerance      relative truncation tolerance for C (in the Frobenius norm)
      //! @param maxRank            maximal rank r' (before enrichment)
      //! @param maxvolTolerance    dominance tolerance for maxvol
      //! @param enrichment         directions to append (r1 x n x rho), an empty block for none; consumed,
      //!                           so pass it with std::move
      //! @param oversampling       relative number of additional rows (0 = square selection)
      //! @param maxEnrichment      maximal number of appended enrichment directions, see enrichFactors
      //! @param maxEnrichedRank    maximal rank r'+rho' after the enrichment (the enrichment is limited accordingly)
      //! @param maxRows            maximal number of selected rows q with oversampling (at least r'+rho' are selected)
      //! @return                   new core (r1 x n x r'+rho'), carry (r'+rho' x r2), selected rows (q) and truncated rank
      //!
      template<typename T>
      CoreUpdate<T> updateCoreLeft(const Tensor3<T>& C, const Tensor2<T>& leftInterface, T rankTolerance, int maxRank, T maxvolTolerance,
                                   Tensor3<T> enrichment = Tensor3<T>(), T oversampling = T(0),
                                   int maxEnrichment = std::numeric_limits<int>::max(),
                                   int maxEnrichedRank = std::numeric_limits<int>::max(),
                                   int maxRows = std::numeric_limits<int>::max());

      //! maxvol column selection of a wide matrix (r x m with m >= r)
      //!
      //! internal::maxvol selects rows of a tall matrix, so this applies it to the adjoint. Used wherever a
      //! right index set is chosen (the right unfolding of a core is wide).
      //!
      //! @param Vt              wide matrix (r x m)
      //! @param maxvolTolerance dominance tolerance for maxvol
      //! @param coefficients    optional output Vt(:,columns)^{-1} Vt (r x m), the identity on the selected columns
      //!                        (with oversampling Vt(:,columns)^+ Vt, nColumns x m)
      //! @param nColumns        number of columns to select, at least r (0 = r: no oversampling), see maxvol
      //! @return                selected columns (size r or nColumns)
      //!
      //! @throws std::invalid_argument if Vt has more rows than columns or is rank-deficient (see maxvol)
      //!
      template<typename T>
      std::vector<int> maxvolColumns(const ConstTensor2View<T>& Vt, T maxvolTolerance, Tensor2<T>* coefficients = nullptr, int nColumns = 0);

      //! Local core update for a right-to-left sweep (mirror image of updateCoreLeft)
      //!
      //! Truncated SVD of the right unfolding of C (r1 x n r2): C ~ US * Vt with orthonormal rows of Vt (r' x n r2);
      //! maxvolColumns on M = Vt (I_n x R^y_{k+1}) selects the columns J_k, q = r' + ceil(oversampling r') of them
      //! (at most maxColumns); core = fold_right(Vt) (right-orthonormal), carry = US (r1 x r').
      //!
      //! There is no enrichment here: only the left-to-right sweeps enrich (see pitts_tensortrain_cross.hpp).
      //!
      //! @param C                  block in the coordinates of y (r1 x n x r2)
      //! @param rightInterface     R^y_{k+1} (r2 x |J_{k+1}|), the 1x1 identity at the right boundary
      //!
      template<typename T>
      CoreUpdate<T> updateCoreRight(const Tensor3<T>& C, const Tensor2<T>& rightInterface, T rankTolerance, int maxRank,
                                    T maxvolTolerance, T oversampling = T(0), int maxColumns = std::numeric_limits<int>::max());

      //! Rows of a block for a cross without cores (error approximation): maxvol on its dominant left singular vectors
      //!
      //! Truncated SVD of the left unfolding (absoluteTolerance in the Frobenius norm, at most nRows directions), then
      //! maxvol on the orthonormal factor, oversampled to nRows rows (rows with the largest leverage scores, see maxvol).
      //! The number of rows is fixed and directions below the tolerance do not influence the selection, so near
      //! round-off the result does not depend on it (a numerical rank would).
      //!
      //! @param B                  block (r1 x n x r2)
      //! @param absoluteTolerance  directions with a tail below it in the Frobenius norm are not dominant
      //! @param nRows              number of rows to select (limited to r1*n)
      //! @param maxvolTolerance    dominance tolerance for maxvol
      //! @return                   selected rows of the left unfolding of B
      //!
      template<typename T>
      std::vector<int> dominantRows(const Tensor3<T>& B, T absoluteTolerance, int nRows, T maxvolTolerance);

      //! Columns of a block for a cross without cores (mirror image of dominantRows, on the right unfolding)
      template<typename T>
      std::vector<int> dominantColumns(const Tensor3<T>& B, T absoluteTolerance, int nColumns, T maxvolTolerance);


      //! Initial y for TT-cross in the orthogonal gauge: zero, but with an orthonormal right frame from the inputs
      //!
      //! The cores 1..d-1 of y are the right-orthonormal cores of the direct sum (block-diagonal cores) of the inputs,
      //! each left-normalized to unit norm first (tolerance 0) and then right-normalized together (tolerance 0, ranks
      //! limited by maxRank and the dimensions on the left). So the right parts of y span the right parts of all
      //! inputs, independent of their norms and gauges (also directions that cancel in their sum); core 0 is set to zero. Then rightIndexSetsFromFrame gives well-conditioned initial right index
      //! sets with full-rank interfaces, as needed by solveRightInterface.
      //!
      //! @param TTx      input tensor trains (same dimensions as TTy, boundary ranks 1)
      //! @param TTy      result, dimensions unchanged
      //! @param maxRank  maximal rank of the frame (the right-normalization truncates the direct sum to it)
      //!
      //! @throws std::invalid_argument for no input, different dimensions or boundary ranks != 1
      //!
      template<typename T>
      void initialRightFrame(const std::vector<std::reference_wrapper<const TensorTrain<T>>>& TTx, TensorTrain<T>& TTy,
                             int maxRank = std::numeric_limits<int>::max());

      //! Right index sets from a maxvol sweep over the right-orthonormal cores 1..d-1 of a tensor train
      //!
      //! Right-to-left: XR_k = X_k * R_{k+1}, J_k = maxvol columns of unfold_right(XR_k) (oversampled: at least the
      //! rank r_{k-1}, r_{k-1} + ceil(oversampling r_{k-1}) if within maxIndices[k-1]), R_k = the selected columns.
      //! With right-orthonormal cores, XR_k keeps full row rank as long as R_{k+1} is a nonsingular (maxvol) submatrix,
      //! so all interfaces have full rank (the QR step of Oseledets/Tyrtyshnikov 2010, reference [1] in
      //! pitts_tensortrain_cross.hpp). Core 0 is not used. No function evaluations are needed.
      //!
      //! @param TT               tensor train with right-orthonormal cores 1..d-1 (e.g. from initialRightFrame)
      //! @param maxvolTolerance  dominance tolerance for maxvol
      //! @param oversampling     relative number of additional columns
      //! @param maxIndices       maximal size of the sets per bond (d-1 entries; empty: no limit)
      //! @return                 right index sets, J[k] for k = 1..d-1 (J[0] empty)
      //!
      template<typename T>
      std::vector<NestedIndices> rightIndexSetsFromFrame(const TensorTrain<T>& TT, T maxvolTolerance, T oversampling = T(0),
                                                         const std::vector<int>& maxIndices = {});
    }
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_cross_helper_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_CROSS_HELPER_HPP
