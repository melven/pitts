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
#include <exception>
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
  void centroids(const MultiVector<T>& X, const std::vector<int>& idx, const std::vector<T>& w, MultiVector<T>& Y)
  {
    const auto chunks = X.rowChunks();
    const auto n = X.cols();
    if( n != idx.size() || n != w.size() )
      throw std::invalid_argument("PITTS::centroids: Dimension mismatch, size of idx and w must match with the number of columns in X!");

    // set Y to zero
    for(int j = 0; j < Y.cols(); j++)
      for(int c = 0; c < chunks; c++)
        Y.chunk(c,j) = Chunk<T>{};

    for(int c = 0; c < chunks; c++)
      for(int i = 0; i < n; i++)
        fmadd(w[i], X.chunk(c,i), Y.chunk(c,idx[i]));
  }
}


#endif // PITTS_MULTIVECTOR_CENTROIDS_HPP
