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
* Index conventions follow the unfoldings in pitts_tensor3_unfold.hpp:
*   left unfolding  (r_{k-1} n_k) x r_k : row    = a + i*r_{k-1}   for (a in I_{k-1}, i in S_k)
*   right unfolding r_{k-1} x (n_k r_k) : column = i + b*n_k       for (i in S_k, b in J_{k+1})
*
* Sketch of one left-to-right step at core k with these helpers (for a TT x and its cross sets I, J):
*
*   normalize_contract1(Lx.left(k-1), X_k, LX);     // LX = L_{k-1} * X_k        (|I_{k-1}| x n_k x r_k(x))
*   normalize_contract2(LX, Lx.right(k+1), B);      // B  = LX * R_{k+1}         (|I_{k-1}| x n_k x |J_{k+1}|)
*   applyToBlock(f, B);                             // B  = f(x) on the cross
*   auto upd = updateCoreLeft(B, tol, maxRank, maxvolTol, enrichment);
*   I[k] = leftIndicesFromRows(upd.selected, |I_{k-1}|);
*   Lx.updateLeft(k, LX, I[k]);                     // L_k = rows I_k of unfold_left(LX)
*   Y_k = upd.core;  Y_{k+1} = upd.carry * Y_{k+1};
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_CROSS_HELPER_HPP
#define PITTS_TENSORTRAIN_CROSS_HELPER_HPP

// includes
#include <vector>
#include "pitts_tensor2.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain.hpp"

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
      //! Only these two numbers per element are needed by the algorithms; full multi-indices
      //! are recovered by following the links (expandLeft / expandRight, for tests and debugging).
      //!
      struct NestedIndices
      {
        //! position in the neighboring (parent) index set: I_{k-1} for left sets, J_{k+1} for right sets
        std::vector<int> link;

        //! mode index i_k of the own dimension
        std::vector<int> idx;

        //! number of multi-indices in the set (= TT rank of the bond)
        [[nodiscard]] int size() const {return static_cast<int>(idx.size());}

        //! row numbers of the multi-indices in a left unfolding with r_prev rows per mode index: link + idx*r_prev
        [[nodiscard]] std::vector<int> rowIndices(int r_prev) const;

        //! column numbers of the multi-indices in a right unfolding with mode size n: idx + link*n
        [[nodiscard]] std::vector<int> columnIndices(int n) const;
      };

      //! decode a row selection of a left unfolding (r_prev*n x r) into a nested left index set
      //!
      //! @param rows     selected rows (e.g. from maxvol), each row = link + idx*r_prev
      //! @param r_prev   number of rows per mode index (= |I_{k-1}|)
      //!
      NestedIndices leftIndicesFromRows(const std::vector<int>& rows, int r_prev);

      //! decode a column selection of a right unfolding (r x n*r_next) into a nested right index set
      //!
      //! @param columns  selected columns (e.g. from maxvol of the transposed unfolding), each column = idx + link*n
      //! @param n        mode size
      //!
      NestedIndices rightIndicesFromColumns(const std::vector<int>& columns, int n);

      //! full multi-indices (i_0,...,i_k) of the left set I_k (I[0] is the set of bond 0)
      std::vector<std::vector<int>> expandLeft(const std::vector<NestedIndices>& I, int k);

      //! full multi-indices (i_k,...,i_{d-1}) of the right set J_k (J[k] is the set of bond k-1, J[0] unused)
      std::vector<std::vector<int>> expandRight(const std::vector<NestedIndices>& J, int k, int nDim);


      //! copy selected rows of a matrix: B = A(rows,:)
      template<typename T>
      void selectRows(const ConstTensor2View<T>& A, const std::vector<int>& rows, Tensor2<T>& B);

      //! copy selected columns of a matrix: B = A(:,columns)
      template<typename T>
      void selectColumns(const ConstTensor2View<T>& A, const std::vector<int>& columns, Tensor2<T>& B);


      //! Cached interface matrices (partial contractions) of one tensor train on nested index sets
      //!
      //! left(k)  = L_k of shape |I_k| x r_k       with L_k(a,:) = X_0(i_0) X_1(i_1) ... X_k(i_k) for (i_0..i_k) = I_k[a]
      //! right(k) = R_k of shape r_{k-1} x |J_k|   with R_k(:,b) = X_k(i_k) ... X_{d-1}(i_{d-1}) for (i_k..i_{d-1}) = J_k[b]
      //! left(-1) and right(nDim) are the 1x1 identity (boundary ranks 1).
      //!
      //! The entries are updated incrementally during sweeps: L_k only depends on L_{k-1}, X_k and I_k,
      //! so after a change of I_k (or X_k) all L_j with j >= k are invalid; analogously for R_k.
      //! Used for x on the index sets of y, for x and y on the index sets of the error approximation z.
      //!
      template<typename T>
      class CrossInterfaces
      {
        public:
          //! set up for a TT with nDim cores, all entries invalid except the boundaries
          explicit CrossInterfaces(int nDim);

          //! number of dimensions
          [[nodiscard]] int nDim() const {return nDim_;}

          //! left interface L_k, k = -1..nDim-1, must be valid
          [[nodiscard]] const Tensor2<T>& left(int k) const;

          //! right interface R_k, k = 0..nDim, must be valid
          [[nodiscard]] const Tensor2<T>& right(int k) const;

          //! is L_k valid?
          [[nodiscard]] bool leftValid(int k) const;

          //! is R_k valid?
          [[nodiscard]] bool rightValid(int k) const;

          //! recompute L_k = rows I_k of unfold_left(LX), where LX = L_{k-1} * X_k (|I_{k-1}| x n_k x r_k) was
          //! computed by the caller (it is needed for the block evaluation anyway); invalidates L_j for j > k
          void updateLeft(int k, const Tensor3<T>& LX, const NestedIndices& Ik);

          //! recompute R_k = columns J_k of unfold_right(XR), where XR = X_k * R_{k+1} (r_{k-1} x n_k x |J_{k+1}|)
          //! was computed by the caller; invalidates R_j for j < k
          void updateRight(int k, const Tensor3<T>& XR, const NestedIndices& Jk);

          //! mark L_k..L_{nDim-1} as invalid (after I_k or X_k changed)
          void invalidateLeft(int k);

          //! mark R_0..R_k as invalid (after J_k or X_k changed)
          void invalidateRight(int k);

        private:
          //! number of dimensions
          int nDim_;

          //! left interfaces, index k+1 for L_k
          std::vector<Tensor2<T>> left_;

          //! right interfaces, index k for R_k
          std::vector<Tensor2<T>> right_;

          //! first invalid left index (L_k valid for k < leftValidUpTo_)
          int leftValidUpTo_;

          //! last invalid right index (R_k valid for k > rightValidFrom_)
          int rightValidFrom_;
      };


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
      //!     rankTolerance on the singular values and maximal rank maxRank;
      //!  2. optional enrichment: the columns of E (r1 n x rho, e.g. error directions from the AMEn-style
      //!     residual) are orthogonalized against U and appended, U' = [U, Q_E] (r1 n x r'+rho'), and SVt is
      //!     padded with zero rows (rho' <= rho, linearly dependent directions are dropped);
      //!  3. maxvol(U', maxvolTolerance) selects the rows I_k;
      //!  4. interpolation gauge: core = fold_left(U' * U'(rows,:)^{-1}), carry = U'(rows,:) * SVt'.
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
      //! @param J                  [out] right index sets, J[k] for k = 1..d-1 (J[0] empty)
      //! @param Rx                 [out] right interfaces of x on J (right(k) valid for k = 1..d)
      //!
      template<typename T>
      void initialRightIndexSets(const TensorTrain<T>& TTx, T maxvolTolerance,
                                 std::vector<NestedIndices>& J, CrossInterfaces<T>& Rx);
    }
  }
}

// TODO(tt-cross): #include "pitts_tensortrain_cross_helper_impl.hpp" unless PITTS_DEVELOP_BUILD once implemented

#endif // PITTS_TENSORTRAIN_CROSS_HELPER_HPP
