// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_triangular_mult.hpp
* @brief in-place matrix multiplication of a tall-skinny matrix and a small upper triangular matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-07-29
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TRIANGULAR_MULT_HPP
#define PITTS_MULTIVECTOR_TRIANGULAR_MULT_HPP

// includes
#include <vector>
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! in-place multiplication of a permuted tall-skinny matrix (multivector) and a small upper triangular matrix (Y <- Y * P * R^T)
  //!
  //! Intended to be used in conjunction with triangularSolve: given a product X Y^T with X P = Q R, we get
  //! X <- X * P * R^(-1)
  //! Y <- Y * P * R^T
  //! => X Y^T stays identical (up to numeric errors), X is almost Q
  //!
  //! @tparam T               underlying data type (double, complex, ...)
  //!
  //! @param X                input and output multi-vector, dimensions (n, m) resized to (n, k) with k <= m on output
  //! @param R                upper triangular matrix, dimensions (k,k)
  //! @param colsPermutation  optional column permutation matrix stored as an array of indices, dimension (k)
  //!
  template<typename T>
  void triangularMult(MultiVector<T>& X, const Tensor2<T>& R, const std::vector<int>& colsPermutation = {});
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_triangular_mult_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_TRIANGULAR_MULT_HPP
