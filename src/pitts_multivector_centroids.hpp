/*! @file pitts_multivector_centroids.hpp
* @brief calculate the weighted sum (centroids) of a set of multi-vectors
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_CENTROIDS_HPP
#define PITTS_MULTIVECTOR_CENTROIDS_HPP

// includes
#include <vector>
#include "pitts_multivector.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the weighted sum of columns in a multi-vector
  //!
  //! @tparam T   underlying data type (double, complex, ...)
  //! @param X    multi-vector  (assumed to have a high number of columns)
  //! @param idx  target column index in Y for each column of X, dimension (X.cols())
  //! @param w    colum weight for each column in Y, dimension (Y.cols())
  //! @param Y    weighted sums of columns of X
  //!
  template<typename T>
  void centroids(const MultiVector<T>& X, const std::vector<long long>& idx, const std::vector<T>& w, MultiVector<T>& Y);
  
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_centroids_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_CENTROIDS_HPP
