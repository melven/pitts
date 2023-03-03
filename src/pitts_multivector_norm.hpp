/*! @file pitts_multivector_norm.hpp
* @brief calculate the 2-norm each vector in a multi-vector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_NORM_HPP
#define PITTS_MULTIVECTOR_NORM_HPP

// includes
#include "pitts_eigen.hpp"
#include "pitts_multivector.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the 2-norm of the columns of a multi-vector
  //!
  //! alpha_i <- sqrt(X(:,i)^T X(:,i))
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param X        input multi-vector, dimensions (n, m)
  //! @return         array of norms (m)
  //!
  template<typename T>
  Eigen::ArrayX<T> norm2(const MultiVector<T>& X);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_norm_impl.hpp"
#endif


#endif // PITTS_MULTIVECTOR_NORM_HPP
