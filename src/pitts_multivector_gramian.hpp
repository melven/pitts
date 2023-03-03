/*! @file pitts_multivector_gramian.hpp
* @brief Calculate the Gram matrix (Gramian) of a multivector (X^T X)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-02-24
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_GRAMIAN_HPP
#define PITTS_MULTIVECTOR_GRAMIAN_HPP

// includes
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the squared distance of each vector in one multi-vector with each vector in another multi-vector
  //!
  //! @tparam T underlying data type (double, complex, ...)
  //! @param X  input multi-vector
  //! @param G  resulting matrix X^T X
  //!
  template<typename T>
  void gramian(const MultiVector<T>& X, Tensor2<T>& G);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_gramian_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_GRAMIAN_HPP
