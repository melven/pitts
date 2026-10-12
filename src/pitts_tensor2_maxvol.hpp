// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor2_maxvol.hpp
* @brief Quasi-maximum-volume row selection (maxvol) for tall matrices, used by TT-cross algorithms
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-11
*
* Reference: Goreinov, Oseledets, Savostyanov, Tyrtyshnikov, Zamarashkin: "How to find a good submatrix",
* in: Matrix Methods: Theory, Algorithms, Applications, World Scientific, 2010, pp. 247-256,
* doi 10.1142/9789812836021_0015.
* Its use in cross approximation of tensor trains: Oseledets, Tyrtyshnikov: "TT-cross approximation for
* multidimensional arrays", Linear Algebra Appl. 432 (1) (2010) 70-88, doi 10.1016/j.laa.2009.07.024.
*
**/

// include guard
#ifndef PITTS_TENSOR2_MAXVOL_HPP
#define PITTS_TENSOR2_MAXVOL_HPP

// includes
#include <vector>
#include "pitts_tensor2.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! Quasi-maximum-volume row selection (maxvol algorithm, Goreinov et al. 2010)
    //!
    //! Selects r rows I of a tall matrix A (m x r, m >= r) such that the submatrix A(I,:) is "dominant",
    //! i.e. all entries of A * A(I,:)^{-1} are bounded by 1+tol in modulus. The volume |det A(I,:)| of a
    //! dominant submatrix is quasi-maximal, and A(I,:) is a good choice for the intersection matrix of
    //! a cross approximation (the interpolation coefficients A * A(I,:)^{-1} stay bounded).
    //!
    //! Algorithm: initial row selection by the pivot rows of an LU decomposition with partial pivoting of A
    //! (with PITTS_DIRECT_MKL_GEMM: ?getrf, BLAS-3, B = A * A(I,:)^{-1} follows with one triangular solve), otherwise
    //! by a column-pivoted QR decomposition of A^T (better initial selection, but unblocked), then
    //! B = A * A(I,:)^{-1}; while max |B_ij| > 1+tol, swap row i into the selection (replacing the j-th
    //! selected row, the volume grows by |B_ij|) and update B with a rank-1 update.
    //!
    //! Oversampling (nRows > r, rectangular maxvol, cf. Mikhalev and Oseledets, Linear Algebra Appl. 538 (2018)
    //! 187-211): afterwards, nRows-r further rows are added greedily, each time the row u that increases the volume
    //! sqrt(det(A(I,:)^H A(I,:))) most, i.e. the one with the largest leverage score u (A(I,:)^H A(I,:))^{-1} u^H.
    //! With C = A L, L L^H = (A(I,:)^H A(I,:))^{-1} (initially C = B), the scores are the squared row norms of C, and
    //! adding the row c = C(i,:) is the symmetric rank-1 update C <- C - beta (C c^H) c with
    //! beta = 1 / (sqrt(1+|c|^2) (1+sqrt(1+|c|^2))) (no inverse or Gram matrix is formed). C is not formed either:
    //! C = B M with a small r x r matrix M that gets the rank-1 updates, the scores are updated with |C c^H|^2/(1+|c|^2),
    //! so each added row costs one product B (M c^H), i.e. O(m r) with only reading B. The interpolation
    //! coefficients are then A A(I,:)^+ = C C(I,:)^H (least squares instead of collocation): their norm, the norm of
    //! the pseudo-inverse of A(I,:) for orthonormal columns of A, only decreases with more rows.
    //!
    //! Complexity: O(m r^2) for the initialization plus O(m r) per swap and per additional row.
    //!
    //! @tparam T         underlying data type (double, complex, ...)
    //!
    //! @param A          tall matrix (m x r, m >= r) with full column rank, e.g. the left unfolding (r1*n x r2) of an orthogonal TT core
    //! @param tol        stop swapping when max |A * A(I,:)^{-1}| <= 1+tol (only the real part is used, must be >= 0)
    //! @param maxIter    maximal number of row swaps (the result is a valid selection even if this limit is hit)
    //! @param AAinv      optional output: the interpolation coefficients A * A(I,:)^+ (m x nRows, A * A(I,:)^{-1} without
    //!                   oversampling, rows I are then the identity), which the algorithm maintains anyway (the core of a
    //!                   TT-cross in the interpolation gauge)
    //! @param nRows      number of rows to select, at least r (0 = r: no oversampling); limited to m
    //! @return           indices of the selected rows I (size min(nRows, m) or r, no duplicates; the first r rows are the
    //!                   square maxvol selection, the additional ones follow in the order they were added)
    //!
    //! @throws std::invalid_argument if m < r, r == 0, 0 < nRows < r, or A is (numerically) rank-deficient
    //!
    template<typename T>
    std::vector<int> maxvol(const ConstTensor2View<T>& A, T tol = T(0.05), int maxIter = 100, Tensor2<T>* AAinv = nullptr, int nRows = 0);
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensor2_maxvol_impl.hpp"
#endif

#endif // PITTS_TENSOR2_MAXVOL_HPP
