// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_cross_helper.hpp
* @brief Building blocks for TT-cross algorithms: nested index sets, interface matrices, core updates
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-11
*
* Notation (see pitts_tensortrain_cross.hpp): a tensor train with d cores X_k of shape r_{k-1} x n_k x r_k,
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
* Lx is a SweepData of interface matrices, see interface_loop_from_left):
*
*   Lx.update(k-1, k+1);                            // L_{k-1} and R_{k+1} valid
*   normalize_contract1(*Lx.left(), X_k, LX);       // LX = L_{k-1} * X_k        (|I_{k-1}| x n_k x r_k(x))
*   normalize_contract2(LX, *Lx.right(), B);        // B  = LX * R_{k+1}         (|I_{k-1}| x n_k x |J_{k+1}|)
*   applyToBlock(f, B);                             // B  = f(x) on the cross
*   auto upd = updateCoreLeft(B, tol, maxRank, maxvolTol, enrichment);
*   I[k] = leftIndicesFromRows(upd.selected, |I_{k-1}|);
*   Lx.invalidate(k);                               // L_k depends on the new I_k, recomputed on the next update
*   Y_k = upd.core;  Y_{k+1} = upd.carry * Y_{k+1};
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_CROSS_HELPER_HPP
#define PITTS_TENSORTRAIN_CROSS_HELPER_HPP

// includes
#include <stdexcept>
#include <utility>
#include <vector>
#include "pitts_tensor2.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_sweep_data.hpp"
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
      //! (Dolgov/Savostyanov 2020, Remark 5) and handled by the sweep order in the driver.
      //!
      struct NestedIndices
      {
        //! position in the neighboring (parent) index set: I_{k-1} for left sets, J_{k+1} for right sets
        std::vector<int> link;

        //! mode index i_k of the own dimension
        std::vector<int> idx;

        //! number of multi-indices in the set (= TT rank of the bond)
        [[nodiscard]] int size() const {return static_cast<int>(idx.size());}
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
        NestedIndices I;
        I.link.resize(rows.size());
        I.idx.resize(rows.size());
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
        NestedIndices J;
        J.link.resize(columns.size());
        J.idx.resize(columns.size());
        for(int b = 0; b < columns.size(); b++)
        {
          if( columns[b] < 0 )
            throw std::invalid_argument("rightIndicesFromColumns: negative column index!");
          J.idx[b] = columns[b] % n;
          J.link[b] = columns[b] / n;
        }
        return J;
      }

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


      //! Interface matrices (partial contractions of a TT on nested index sets), cached with SweepData
      //!
      //! Use with defineSweepData<Tensor2<T>>(nDim, interface_loop_from_left<T>(TT, I), interface_loop_from_right<T>(TT, J)):
      //!   subTensor(k) from the left  = L_k of shape |I_k| x r_k       with L_k(a,:) = X_0(i_0) X_1(i_1) ... X_k(i_k) for (i_0..i_k) = I_k[a]
      //!   subTensor(k) from the right = R_k of shape r_{k-1} x |J_k|   with R_k(:,b) = X_k(i_k) ... X_{d-1}(i_{d-1}) for (i_k..i_{d-1}) = J_k[b]
      //! The boundary interfaces L_{-1} and R_d are the 1x1 identity and appear as std::nullopt (SweepData::left()/right()).
      //!
      //! The recursion L_k(a,:) = L_{k-1}(link[a],:) * X_k(:,idx[a],:) is why the index sets are stored as (link, idx).
      //! L_k is computed by contracting L_{k-1} with the full core (normalize_contract1, all mode indices) and
      //! picking the rows (link, idx) of the result. This is the same contraction the block evaluation of
      //! core k needs, so it is currently done twice per core (once here, once for the block); acceptable for
      //! the first version, a shared scratch tensor can remove it later.
      //!
      //! SweepData's invalidation matches the dependency structure: L_k depends on L_{k-1}, X_k and I_k only,
      //! so after I_k (or X_k) changed, invalidate(k) marks L_k..L_{d-1} invalid and update() recomputes on demand.
      //! Used for x on the index sets of y, and for x and y on the index sets of the error approximation z.
      //!
      //! @param TT   the tensor train (captured by reference, must outlive the SweepData)
      //! @param I    the left index sets (captured by reference, I[k] must be valid when L_k is computed)
      //! @return     lambda (int iDim, optional_cref<Tensor2<T>> prev, Tensor2<T>& result) for defineSweepData
      //!
      template<typename T>
      auto interface_loop_from_left(const TensorTrain<T>& TT, const std::vector<NestedIndices>& I);

      //! right interface matrices for SweepData, see interface_loop_from_left
      //!
      //! @param TT   the tensor train (captured by reference)
      //! @param J    the right index sets (captured by reference, J[k] must be valid when R_k is computed)
      //!
      template<typename T>
      auto interface_loop_from_right(const TensorTrain<T>& TT, const std::vector<NestedIndices>& J);


      //! apply a scalar function to every entry of a block (the only place where f is called)
      //!
      //! @tparam Function  scalar functor T f(T)
      //! @return           number of function evaluations (= number of entries)
      //!
      template<typename T, typename Function>
      long long applyToBlock(const Function& f, Tensor3<T>& B);


      //! relative Frobenius-norm difference ||A - B||_F / ||B||_F of two blocks of equal shape (convergence measure)
      template<typename T>
      auto relativeDifference(const Tensor3<T>& A, const Tensor3<T>& B);


      //! Result of a local core update (updateCoreLeft / updateCoreRight)
      template<typename T>
      struct CoreUpdate
      {
        //! new core in interpolation gauge: r1 x n x r' (left) or r' x n x r2 (right)
        Tensor3<T> core;

        //! remainder to be multiplied into the neighboring core: r' x r2 (left, Y_{k+1} <- carry * Y_{k+1})
        //! or r1 x r' (right, Y_{k-1} <- Y_{k-1} * carry)
        Tensor2<T> carry;

        //! selected rows of the left unfolding (left) or columns of the right unfolding (right), size r'
        std::vector<int> selected;

        //! rank r' after truncation (without enrichment) -- for convergence/rank reporting
        int truncatedRank = 0;
      };

      //! Local core update for a left-to-right sweep
      //!
      //! Given the evaluated block B (r1 x n x r2) of f(x) on the cross (I_{k-1}, S_k, J_{k+1}):
      //!  1. truncated SVD of the left unfolding of B: B ~ U * SVt with U (r1 n x r'), relative tolerance
      //!     rankTolerance on the singular values and maximal rank maxRank (internal::normalize_svd);
      //!  2. optional enrichment: the columns of E (r1 n x rho, e.g. error directions from the AMEn-style
      //!     residual) are orthogonalized against U and appended, U' = [U, Q_E] (r1 n x r'+rho'), and SVt is
      //!     padded with zero rows (rho' <= rho, linearly dependent directions are dropped). This is the
      //!     same sequence as in the enrichSubspace lambda of solveMALS (project out, normalize_qb/svd of the
      //!     remainder, concatLeftRight / concatTopBottom with zero padding) and reuses those building blocks;
      //!  3. maxvol(U', maxvolTolerance) selects the rows I_k;
      //!  4. interpolation gauge: core = fold_left(U' * U'(rows,:)^{-1}), carry = U'(rows,:) * SVt'.
      //!     (maxvol forms U' * U'(rows,:)^{-1} internally; it should return it instead of recomputing it here.)
      //!
      //! Compared to solveMALS this replaces "solve the local problem, then orthogonalize" by "evaluate the
      //! block, then select interpolation rows": the orthogonal gauge and the setSubTensors orthogonality
      //! flags of MALS are not used, y is in the interpolation gauge instead (see pitts_tensortrain_cross.hpp).
      //!
      //! Invariants: unfold_left(core)(rows,:) = identity, |entries of core| <= 1 + maxvolTolerance,
      //! and core * carry = U * SVt (= B up to the truncation error).
      //!
      //! @param B                  evaluated block (r1 x n x r2)
      //! @param rankTolerance      relative truncation tolerance for the singular values of B
      //! @param maxRank            maximal rank r' (before enrichment)
      //! @param maxvolTolerance    dominance tolerance for maxvol
      //! @param enrichment         optional directions to append (r1 n x rho), nullptr for none
      //! @return                   new core, carry, selected rows and truncated rank, see CoreUpdate
      //!
      template<typename T>
      CoreUpdate<T> updateCoreLeft(const Tensor3<T>& B, T rankTolerance, int maxRank, T maxvolTolerance,
                                   const Tensor2<T>* enrichment = nullptr);

      //! Local core update for a right-to-left sweep (mirror image of updateCoreLeft)
      //!
      //! Truncated SVD of the right unfolding of B (r1 x n r2): B ~ US * Vt with Vt (r' x n r2); optional
      //! enrichment rows E (rho x n r2) appended to Vt; maxvol on the transposed Vt' selects the columns J_k;
      //! core = fold_right(Vt'(:,columns)^{-1} * Vt'), carry = US' * Vt'(:,columns) (r1 x r'+rho').
      //!
      //! Invariants: unfold_right(core)(:,columns) = identity, |entries of core| <= 1 + maxvolTolerance,
      //! and carry * core = US * Vt.
      //!
      template<typename T>
      CoreUpdate<T> updateCoreRight(const Tensor3<T>& B, T rankTolerance, int maxRank, T maxvolTolerance,
                                    const Tensor2<T>* enrichment = nullptr);


      //! Initial right index sets and right interfaces from the tensor train x itself (no function evaluations)
      //!
      //! Right-to-left maxvol sweep over a right-orthogonalized copy of x: for k = d-1..1,
      //! XR = X_k * R_{k+1}, J_k = maxvol columns of unfold_right(XR), R_k = the selected columns.
      //! The resulting sets are right-nested with |J_k| = r_{k-1}(x) and the columns of R_k form a
      //! dominant submatrix, so the first left-to-right sweep starts from well-conditioned crosses.
      //!
      //! @param TTx                input tensor train (boundary ranks 1)
      //! @param maxvolTolerance    dominance tolerance for maxvol
      //! @return                   right index sets, J[k] for k = 1..d-1 (J[0] empty); the interfaces are then
      //!                           obtained with a SweepData built from interface_loop_from_right
      //!
      template<typename T>
      std::vector<NestedIndices> initialRightIndexSets(const TensorTrain<T>& TTx, T maxvolTolerance);
    }
  }
}

// TODO(tt-cross): #include "pitts_tensortrain_cross_helper_impl.hpp" unless PITTS_DEVELOP_BUILD once implemented

#endif // PITTS_TENSORTRAIN_CROSS_HELPER_HPP
