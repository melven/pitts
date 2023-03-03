/*! @file pitts_multivector_dot.hpp
* @brief calculate the dot products of each vector in a multi-vector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_DOT_HPP
#define PITTS_MULTIVECTOR_DOT_HPP

// includes
#include "pitts_eigen.hpp"
#include "pitts_multivector.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the pair-wise dot products of the columns of two multi-vectors
  //!
  //! alpha_i <- X(:,i)^T Y(:,i)
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param Y        input multi-vector, dimensions (n, m)
  //! @return         array of dot products
  //!
  template<typename T>
  Eigen::ArrayX<T> dot(const MultiVector<T>& X, const MultiVector<T>& Y);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_dot_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_DOT_HPP
