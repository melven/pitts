// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor2_maxvol.hpp
* @brief Quasi-maximum-volume row selection (maxvol) for tall matrices, used by TT-cross algorithms
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-11
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
    //! Algorithm: initial row selection by a column-pivoted QR decomposition of A^T, then
    //! B = A * A(I,:)^{-1}; while max |B_ij| > 1+tol, swap row i into the selection (replacing the j-th
    //! selected row, the volume grows by |B_ij|) and update B with a rank-1 update.
    //!
    //! Complexity: O(m r^2) for the initialization plus O(m r) per swap.
    //!
    //! @tparam T         underlying data type (double, complex, ...)
    //!
    //! @param A          tall matrix (m x r, m >= r) with full column rank, e.g. the left unfolding (r1*n x r2) of an orthogonal TT core
    //! @param tol        stop swapping when max |A * A(I,:)^{-1}| <= 1+tol (only the real part is used, must be >= 0)
    //! @param maxIter    maximal number of row swaps (the result is a valid selection even if this limit is hit)
    //! @return           indices of the selected rows I (size r, arbitrary order, no duplicates)
    //!
    //! @throws std::invalid_argument if m < r, r == 0, or A is (numerically) rank-deficient
    //!
    template<typename T>
    std::vector<int> maxvol(const ConstTensor2View<T>& A, T tol = T(0.05), int maxIter = 100);
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensor2_maxvol_impl.hpp"
#endif

#endif // PITTS_TENSOR2_MAXVOL_HPP
