/*! @file pitts_multivector_axpby.hpp
* @brief calculate the scaled addition of the pair-wise columns of two multi-vectors
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_AXPBY_HPP
#define PITTS_MULTIVECTOR_AXPBY_HPP

// includes
#include <array>
#include "pitts_eigen.hpp"
#include "pitts_multivector.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the pair-wise scaled addition of the columns of two multi-vectors
  //!
  //! Calculates Y(:,i) <- alpha_i*X(:,i) + Y(:,i)
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param alpha    array of scaling factors, dimension (m)
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param Y        resulting multi-vector, dimensions (n, m)
  //!
  template<typename T>
  void axpy(const Eigen::ArrayX<T>& alpha, const MultiVector<T>& X, MultiVector<T>& Y);
  

  //! calculate the pair-wise scaled addition of the columns of two multi-vectors, and the norm of the result
  //!
  //! Calculates Y(:,i) <- alpha_i*X(:,i) + Y(:,i)
  //!            gamma_i <- ||Y(:,i)||_2
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param alpha    array of scaling factors, dimension (m)
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param Y        resulting multi-vector, dimensions (n, m)
  //! @returns        array of norms of the resulting y, dimension (m)
  //!
  template<typename T>
  Eigen::ArrayX<T> axpy_norm2(const Eigen::ArrayX<T>& alpha, const MultiVector<T>& X, MultiVector<T>& Y);
  

  //! calculate the pair-wise scaled addition of the columns of two multi-vectors, and the dot product of the result with the columsn of a third multi-vector
  //!
  //! Calculates Y(:,i) <- alpha_i*X(:,i) + Y(:,i)
  //!            gamma_i <- Y(:,i)^T * Z(:,i)
  //!            
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param alpha    array of scaling factors, dimension (m)
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param Y        resulting multi-vector, dimensions (n, m)
  //! @param Z        third multi-vector, dimensions (n,m)
  //! @returns        array of dot products between Y and Z, dimension (m)
  //!
  template<typename T>
  Eigen::ArrayX<T> axpy_dot(const Eigen::ArrayX<T>& alpha, const MultiVector<T>& X, MultiVector<T>& Y, const MultiVector<T>& Z);
  
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_axpby_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_AXPBY_HPP
