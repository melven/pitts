/*! @file pitts_multivector_cdist.hpp
* @brief calculate the distance of each vector in one multi-vector with each vector in another multi-vector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_CDIST_HPP
#define PITTS_MULTIVECTOR_CDIST_HPP

// includes
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the squared distance of each vector in one multi-vector with each vector in another multi-vector
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @param X  first multi-vector  (assumed to have a high number of columns)
  //! @param Y  second multi-vector (assumed to have a low number of columns)
  //! @param D  pair-wise squared distance of each column in X and Y, dimension (X.cols() x Y.rows())
  //!
  template<typename T>
  void cdist2(const MultiVector<T>& X, const MultiVector<T>& Y, Tensor2<T>& D);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_cdist_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_CDIST_HPP
