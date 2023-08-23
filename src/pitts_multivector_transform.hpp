/*! @file pitts_multivector_transform.hpp
* @brief calculate the matrix product of a tall-skinny matrix and a small matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-07-30
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TRANSFORM_HPP
#define PITTS_MULTIVECTOR_TRANSFORM_HPP

// includes
#include <array>
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the matrix-matrix product of a tall-skinny matrix (multivector) with a small matrix (Y <- X*M)
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param M        small transformation matrix, dimensions (m, k)
  //! @param Y        resulting mult-vector, resized to dimensions (n, k) or desired shape (see below)
  //! @param reshape  desired shape of the resulting multi-vector, total size must be n*k
  //!
  template<typename T>
  void transform(const MultiVector<T>& X, const ConstTensor2View<T>& M, MultiVector<T>& Y, std::array<long long,2> reshape = {0, 0});

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_transform_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_TRANSFORM_HPP
