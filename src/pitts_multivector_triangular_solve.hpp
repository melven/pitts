/*! @file pitts_multivector_triangular_solve.hpp
* @brief in-place triangular solve (backward substitution) with a tall-skinny matrix and a small upper triangular matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-03-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TRIANGULAR_SOLVE_HPP
#define PITTS_MULTIVECTOR_TRIANGULAR_SOLVE_HPP

// includes
#include <vector>
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! in-place triangular solve with a permuted tall-skinny matrix (multivector) and a small upper triangular matrix (X <- X*R^(-1)*P^T)
  //!
  //! @tparam T               underlying data type (double, complex, ...)
  //!
  //! @param X                input and output multi-vector, dimensions (n, m) resized to (n, k) with k <= m on output
  //! @param R                upper triangular matrix, dimensions (k,k)
  //! @param colsPermutation  optional column permutation matrix stored as an array of indices, dimension (k)
  //!
  template<typename T>
  void triangularSolve(MultiVector<T>& X, const Tensor2<T>& R, const std::vector<int>& colsPermutation = {});

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_triangular_solve_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_TRIANGULAR_SOLVE_HPP
