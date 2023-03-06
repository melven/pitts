/*! @file pitts_tensor2_qb_decomposition.hpp
* @brief QB-part of the SVQB orthogonalization algorithm (Stathopoulos and Wu, SISC 23 (6), pp. 2165-2182)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-31
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR2_QB_DECOMPOSITION_HPP
#define PITTS_TENSOR2_QB_DECOMPOSITION_HPP

// includes
#include "pitts_tensor2.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! QB-part in the SVQB orthogonalization algorithm from Stathopoulos and Wu, SISC 23 (6), pp. 2165-2182
  //!
  //! Computes the decomposition B^TB = M using a SVD of M for a symmetric positive semi-definite matrix M
  //!
  //! @tparam T                 underlying data type (double, complex, ...)
  //!
  //! @param  M                 Symmetric positive semi-definite input matrix M
  //! @param  B                 Resulting matrix B with B^TB = M
  //! @param  Binv              Pseudo-Inverse of the output matrix B
  //! @param  rankTolerance     Desired accuracy / truncation tolerance
  //! @param  absoluteTolerance Use rankTolerance as absolute truncation tolerance instead of relative to the biggest singular value.
  //! @return                   detected rank of the matrix
  //!
  template<typename T>
  int qb_decomposition(const Tensor2<T>& M, Tensor2<T>& B, Tensor2<T>& Binv, T rankTolerance,  int maxRank = std::numeric_limits<int>::max(), bool absoluteTolerance = false);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensor2_qb_decomposition_impl.hpp"
#endif

#endif // PITTS_TENSOR2_QB_DECOMPOSITION_HPP
