// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_symmetric.hpp
* @brief Operations on symmetric tensors y = G S G^T given by a left half G (tensor train with an open right bond)
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-10-20
*
* A symmetric tensor with d = 2m dimensions and n_k = n_{d-1-k},
*   y(i_0, ..., i_{2m-1}) = G(i_0, ..., i_{m-1}, :) * S * G(i_{2m-1}, ..., i_m, :)^T,
* is stored by its left half G, a tensor train with m cores whose last core has an open right bond of size r
* (the rank at the middle bond), and a symmetric r x r matrix S. Seen as a matrix with the rows (i_0, ..., i_{m-1})
* and the mirrored columns (i_{2m-1}, ..., i_m), y = A S A^T with the N x r matrix A of the values of G (N = n_0 ... n_{m-1}).
* The special case S = I, y = H H^T, is a positive semidefinite tensor given by its half H.
*
* Orthogonal gauge: with G left-orthonormal (all m cores, including the last one, A^T A = I), y = A S A^T is an
* eigendecomposition-like factorization: the singular values of y at the middle bond are those of S (the absolute values
* of its eigenvalues), at an inner bond k < m-1 those of the right part [G_{k+1} ... G_{m-1}] S of the half (the mirrored
* G^T has orthonormal rows). This makes the rounding below a truncation of y itself with a rigorous error bound.
*
* Only real types: the symmetric (not Hermitian) eigenvalue problem of a complex symmetric S is not unitarily diagonalizable.
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SYMMETRIC_HPP
#define PITTS_TENSORTRAIN_SYMMETRIC_HPP

// includes
#include <limits>
#include <vector>
#include "pitts_tensor2.hpp"
#include "pitts_tensortrain.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! operations on symmetric tensors G S G^T given by a left half G, see pitts_tensortrain_symmetric.hpp
    namespace symmetric_tt
    {
      //! Left-orthonormalize all cores of a half, including the last one with the open right bond
      //!
      //! H = G R on the open bond: afterwards H holds G (left-orthonormal, last core r_{m-1} x n_{m-1} x r'), so the old half is
      //! H R. Tolerance 0: only numerically dependent directions are removed (r' <= r, as in leftNormalize).
      //!
      //! @param H  half with m cores (first rank 1, open last rank r), overwritten by G
      //! @return   R (r' x r)
      //!
      template<typename T>
      Tensor2<T> leftOrthonormalizeHalf(TensorTrain<T>& H);

      //! The matrix A^T B of two halves with the same dimensions (rA x rB, the inner products of the columns over all entries)
      //!
      //! Contraction from the left to the right (as in dot): E_{k+1} = sum_i A_k(:,i,:)^T E_k B_k(:,i,:), E_0 = 1.
      //!
      //! @param A  half with m cores and open last rank rA
      //! @param B  half with the same dimensions and open last rank rB
      //! @return   A^T B (rA x rB)
      //!
      template<typename T>
      Tensor2<T> halfInnerProducts(const TensorTrain<T>& A, const TensorTrain<T>& B);

      //! The half G S (last core times S) with right-orthonormal cores 1..m-1 (tolerance 0, core 0 holds the rest)
      //!
      //! With a left-orthonormal G, the mirrored G^T has orthonormal rows, so y = G S G^T = Y G^T with right-orthonormal parts
      //! Y_{k+1..m-1} G^T at every bond k: the orthogonal gauge of y seen from the left (used by symmetricRound, and as the
      //! frame of the symmetric TT-cross).
      //!
      //! @param G   half (m cores, open last rank r)
      //! @param S   r x r matrix
      //! @param Y   result (same dimensions as G)
      //!
      template<typename T>
      void rightOrthonormalHalf(const TensorTrain<T>& G, const Tensor2<T>& S, TensorTrain<T>& Y);

      //! (Truncated) eigendecomposition C ~ U diag(lambda) U^T of a real symmetric matrix
      template<typename T>
      struct SymmetricEigen
      {
        //! orthonormal eigenvectors (n x r)
        Tensor2<T> U;

        //! eigenvalues, sorted by decreasing absolute value
        std::vector<T> lambda;
      };

      //! Truncated eigendecomposition of a real symmetric matrix, sorted by decreasing absolute value
      //!
      //! The input is symmetrized first ((C + C^T)/2, the Frobenius-nearest symmetric matrix: it only removes the
      //! asymmetry from round-off of a matrix that is symmetric in exact arithmetic). The singular values of a symmetric
      //! matrix are the absolute values of its eigenvalues, so the truncation in the Frobenius norm as in
      //! internal::normalize_svd keeps the largest |lambda|: the discarded ones satisfy sqrt(sum lambda_j^2) <= tolerance
      //! (relative to ||C||_F unless absoluteTolerance), and at most maxRank are kept (at least one).
      //!
      //! @param C                  symmetric matrix (n x n)
      //! @param rankTolerance      truncation tolerance in the Frobenius norm (0: keep all)
      //! @param maxRank            maximal number of eigenpairs
      //! @param absoluteTolerance  rankTolerance is absolute instead of relative to ||C||_F
      //! @return                   eigenvectors and eigenvalues
      //!
      //! @throws std::invalid_argument if C is not square
      //!
      template<typename T>
      SymmetricEigen<T> symmetricEigen(const Tensor2<T>& C, T rankTolerance = T(0), int maxRank = std::numeric_limits<int>::max(),
                                       bool absoluteTolerance = false);

      //! TT-rounding of a symmetric tensor y = G S G^T, keeping the symmetric form (orthogonal gauge on output)
      //!
      //! With G left-orthonormal on input:
      //!  1. the half T = G S (last core times S) is right-orthonormalized (tolerance 0) and truncated by a left-to-right SVD
      //!     sweep over the inner bonds (as in PITTS::normalize). The result is T' = G' M with G' left-orthonormal and
      //!     M = G'^T T, and ||T - T'||_F <= tau sqrt(m-1) ||y||_F (y = T G^T, G^T has orthonormal rows).
      //!  2. symmetric projection: with the projector P onto span(G'), y'' = P y P = G' S'' G'^T, S'' = M W^T and
      //!     W = G'^T G. y - y'' = (I-P) y + P y (I-P), two Frobenius-orthogonal terms of norm <= ||(I-P) y|| <= ||T - T'||
      //!     each (y is symmetric), so ||y - y''|| <= sqrt(2) ||T - T'|| (the truncation of an inner bond acts on both halves).
      //!  3. truncated eigendecomposition of S'' (the middle bond): S' = diag(lambda), G'_{m-1} <- G'_{m-1} U.
      //!     This error lies in span(G') x span(G') and is orthogonal to the one of step 2.
      //! With the tolerance tau per bond, the squared errors add up to at most (2(m-1) + 1) tau^2 ||y||^2 (an inner bond of the
      //! half is two bonds of y), so tau = rankTolerance / sqrt(d-1) with d = 2m, the per-bond tolerance of PITTS::normalize
      //! for y, gives ||y - y'||_F <= rankTolerance ||y||_F.
      //!
      //! @param G              left-orthonormal half (m cores, open last rank r), overwritten by the left-orthonormal G'
      //! @param S              symmetric r x r matrix, overwritten by the diagonal S' (eigenvalues sorted by decreasing |lambda|)
      //! @param rankTolerance  relative truncation tolerance for y in the Frobenius norm
      //! @param maxRank        maximal rank at each bond (inner and middle)
      //!
      //! @throws std::invalid_argument for mismatching dimensions
      //!
      template<typename T>
      void symmetricRound(TensorTrain<T>& G, Tensor2<T>& S, T rankTolerance, int maxRank = std::numeric_limits<int>::max());

      //! Positive semidefinite half H with H H^T = G S_+ G^T, where S_+ is S without its negative eigenvalues
      //!
      //! S_+ is the Frobenius-nearest positive semidefinite matrix to S; with G left-orthonormal, G S_+ G^T is the
      //! nearest positive semidefinite tensor to y = G S G^T in the subspace span(G) x span(G), with the error
      //! ||lambda_-||_2. Only the eigenvalues > 0 are kept (H has the rank of S_+; if there is none, H is zero with rank 1).
      //!
      //! @param G   half (m cores, open last rank r)
      //! @param S   symmetric r x r matrix
      //! @param H   result: G with the last core times U_+ diag(sqrt(lambda_+))
      //! @return    ||lambda_-||_2, the Frobenius norm of the removed negative part (relative to ||y|| only for left-orthonormal G)
      //!
      template<typename T>
      T positiveHalf(const TensorTrain<T>& G, const Tensor2<T>& S, TensorTrain<T>& H);
    }
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_symmetric_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_SYMMETRIC_HPP
