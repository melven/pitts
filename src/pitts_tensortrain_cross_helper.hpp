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
*   auto upd = updateCoreLeft(B, tol, maxRank, maxvolTol, enrichment);
*   I[k] = leftIndicesFromRows(upd.selected, |I_{k-1}|);
*   Lx.invalidate(k+1);                             // LX_{k+1} depends on the new I_k, recomputed on the next update
*   Y_k = upd.core;  Y_{k+1} = upd.carry * Y_{k+1};
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_CROSS_HELPER_HPP
#define PITTS_TENSORTRAIN_CROSS_HELPER_HPP

// includes
#include <algorithm>
#include <cassert>
#include <concepts>
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
      //! @param B     block to contract (m x n x r_{k-1})
      //! @param XR    cached contraction XR_k of the neighboring dimension, nullopt at the right boundary
      //! @param J     the right index sets
      //! @param k     dimension of XR: J[k] (the set of bond k-1) selects its columns; pass iDim+1 for a
      //!              block ending at core iDim (unused at the boundary)
      //! @param C     result (m x n x |J_k|)
      //!
      template<typename T>
      void contractRightInterface(const Tensor3<T>& B, optional_cref<Tensor3<T>> XR, const std::vector<NestedIndices>& J, int k,
                                  Tensor3<T>& C)
      {
        if( XR )
          normalize_contract2(B, interfaceRight(XR->get(), J.at(k)), C);
        else
          copy(B, C);
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
      //! @param TT   the tensor train (captured by reference)
      //! @param J    the right index sets (captured by reference, J[k+1] must be valid when XR_k is computed)
      //!
      template<typename T>
      auto contraction_loop_from_right(const TensorTrain<T>& TT, const std::vector<NestedIndices>& J)
      {
        return [&](int iDim, optional_cref<Tensor3<T>> prev, Tensor3<T>& XR)
        {
          // XR_iDim = X_iDim R_{iDim+1} with R_{iDim+1} selected from the previous contraction
          contractRightInterface<T>(TT.subTensor(iDim), prev, J, iDim+1, XR);
        };
      }


      //! the cores iDim..iDim+nMALS-1 of a tensor train combined into a single block (modes combined as in combine)
      //!
      //! For y in the interpolation gauge this is y on its own cross (both interfaces are the identity there).
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
      //! @param J       right index sets
      //! @param B       block (|I_{iDim-1}| x n_iDim...n_{iDim+nMALS-1} x |J_{iDim+nMALS}|, modes combined as in combine)
      //!
      template<typename T>
      void localBlockFromLeft(const Tensor3<T>& LX, const TensorTrain<T>& TTx, int iDim, int nMALS,
                              optional_cref<Tensor3<T>> XR, const std::vector<NestedIndices>& J, Tensor3<T>& B)
      {
        if( nMALS == 1 )
          return contractRightInterface<T>(LX, XR, J, iDim+1, B);

        contractRightInterface<T>(combine(LX, combineCores(TTx, iDim+1, nMALS-1)), XR, J, iDim+nMALS, B);
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
      //! @tparam Function  scalar functor T f(T) or array functor f(MultiVector<T>&)
      //! @return           number of function evaluations (= number of entries)
      //!
      //! @throws std::invalid_argument if an array function changes the dimensions of the block
      //!
      template<typename T, typename Function>
      long long applyToBlock(const Function& f, Tensor3<T>& B)
      {
        const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

        const auto r1 = B.r1();
        const auto n = B.n();
        const auto r2 = B.r2();

        if constexpr( std::invocable<const Function&, MultiVector<T>&> )
        {
          MultiVector<T> entries = unfold(std::move(B));
          f(entries);
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
                B(i, j, k) = f(B(i, j, k));
        }

        return r1*n*r2;
      }


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
      //!
      //! @param Q   orthogonal factor as a core, extended in place
      //! @param B   remainder, padded with zero rows in place
      //! @param E   candidate directions as a core, consumed (the projection overwrites it)
      //!
      template<typename T>
      void enrichFactors(Tensor3<T>& Q, Tensor2<T>& B, Tensor3<T>&& E);


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
      //!  1. truncated SVD of the left unfolding of B: B ~ U * SVt with U (r1 n x r'), truncation in the
      //!     Frobenius norm with relative tolerance rankTolerance and maximal rank maxRank
      //!     (internal::normalize_svd, same criterion as PITTS::normalize);
      //!  2. optional enrichment: the block E (r1 x n x rho, e.g. error directions from the AMEn-style
      //!     residual) is orthogonalized against U and appended, U' = [U, Q_E] (r1 n x r'+rho'), and SVt is
      //!     padded with zero rows (rho' <= rho: directions in the span of U and those beyond the dimension
      //!     of its orthogonal complement are dropped, see enrichFactors). This is the same sequence as in
      //!     the enrichSubspace lambda of solveMALS and reuses the same contraction kernels;
      //!  3. maxvol(U', maxvolTolerance) selects the rows I_k;
      //!  4. interpolation gauge: core = fold_left(U' * U'(rows,:)^{-1}) (returned by maxvol), carry = U'(rows,:) * SVt'.
      //!
      //! Compared to solveMALS this replaces "solve the local problem, then orthogonalize" by "evaluate the
      //! block, then select interpolation rows": the orthogonal gauge and the setSubTensors orthogonality
      //! flags of MALS are not used, y is in the interpolation gauge instead (see pitts_tensortrain_cross.hpp).
      //!
      //! Invariants: unfold_left(core)(rows,:) = identity, |entries of core| <= 1 + maxvolTolerance,
      //! and core * carry = U * SVt (= B up to the truncation error).
      //!
      //! @param B                  evaluated block (r1 x n x r2)
      //! @param rankTolerance      relative truncation tolerance for B (in the Frobenius norm)
      //! @param maxRank            maximal rank r' (before enrichment)
      //! @param maxvolTolerance    dominance tolerance for maxvol
      //! @param enrichment         directions to append (r1 x n x rho), an empty block for none; consumed,
      //!                           so pass it with std::move
      //! @return                   new core, carry, selected rows and truncated rank, see CoreUpdate
      //!
      template<typename T>
      CoreUpdate<T> updateCoreLeft(const Tensor3<T>& B, T rankTolerance, int maxRank, T maxvolTolerance,
                                   Tensor3<T> enrichment = Tensor3<T>());

      //! maxvol column selection of a wide matrix (r x m with m >= r)
      //!
      //! internal::maxvol selects rows of a tall matrix, so this applies it to the adjoint. Used wherever a
      //! right index set is chosen (the right unfolding of a core is wide).
      //!
      //! @param Vt              wide matrix (r x m)
      //! @param maxvolTolerance dominance tolerance for maxvol
      //! @param coefficients    optional output Vt(:,columns)^{-1} Vt (r x m), the identity on the selected columns
      //! @return                selected columns (size r)
      //!
      //! @throws std::invalid_argument if Vt has more rows than columns or is rank-deficient (see maxvol)
      //!
      template<typename T>
      std::vector<int> maxvolColumns(const ConstTensor2View<T>& Vt, T maxvolTolerance, Tensor2<T>* coefficients = nullptr);

      //! Local core update for a right-to-left sweep (mirror image of updateCoreLeft)
      //!
      //! Truncated SVD of the right unfolding of B (r1 x n r2): B ~ US * Vt with Vt (r' x n r2);
      //! maxvolColumns(Vt) selects the columns J_k; core = fold_right(Vt(:,columns)^{-1} * Vt),
      //! carry = US * Vt(:,columns) (r1 x r').
      //!
      //! There is no enrichment here: only the left-to-right sweeps enrich (see pitts_tensortrain_cross.hpp).
      //!
      //! Invariants: unfold_right(core)(:,columns) = identity, |entries of core| <= 1 + maxvolTolerance,
      //! and carry * core = US * Vt.
      //!
      template<typename T>
      CoreUpdate<T> updateCoreRight(const Tensor3<T>& B, T rankTolerance, int maxRank, T maxvolTolerance);


      //! Initial right index sets and right interfaces from the tensor train x itself (no function evaluations)
      //!
      //! Right-to-left maxvol sweep over a right-orthogonalized copy of x: for k = d-1..1,
      //! XR = X_k * R_{k+1}, J_k = maxvol columns of unfold_right(XR), R_k = the selected columns.
      //! The resulting sets are right-nested with |J_k| = r_{k-1}(x) (after removing exactly dependent
      //! ranks) and the columns of R_k form a dominant submatrix in the orthogonalized TT, so the first
      //! left-to-right sweep starts from well-conditioned crosses. No function evaluations are needed.
      //!
      //! @param TTx                input tensor train (boundary ranks 1)
      //! @param maxvolTolerance    dominance tolerance for maxvol
      //! @return                   right index sets, J[k] for k = 1..d-1 (J[0] empty); the interfaces are then
      //!                           obtained with a SweepData built from contraction_loop_from_right
      //!
      template<typename T>
      std::vector<NestedIndices> initialRightIndexSets(const TensorTrain<T>& TTx, T maxvolTolerance);
    }
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_cross_helper_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_CROSS_HELPER_HPP
