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
#include <stdexcept>
#include "pitts_multivector.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

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
  void centroids(const MultiVector<T>& X, const std::vector<long long>& idx, const std::vector<T>& w, MultiVector<T>& Y)
  {
    const auto chunks = X.rowChunks();
    const auto n = X.cols();
    const auto m = Y.cols();
    if( n != idx.size() || n != w.size() )
      throw std::invalid_argument("PITTS::centroids: Dimension mismatch, size of idx and w must match with the number of columns in X!");

    // gather performance data
    const double rowsd = X.rows();
    const double nd = n;
    const double md = m;
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"rows","Xcols","Ycols"},{X.rows(),n,m}}, // arguments
        {{(rowsd*nd)*kernel_info::FMA<T>()}, // flops
         {(nd*rowsd+nd)*kernel_info::Load<T>() + nd*kernel_info::Load<long long>() + (md*rowsd)*kernel_info::Store<T>()}});


#pragma omp parallel
    {
      for(long long j = 0; j < m; j++)
      {
#pragma omp for schedule(static) nowait
        for(long long c = 0; c < chunks; c++)
	{
          Y.chunk(c,j) = Chunk<T>{};
	}
      }

      for(long long i = 0; i < n; i++)
      {
#pragma omp for schedule(static) nowait
        for(long long c = 0; c < chunks; c++)
          fmadd(w[i], X.chunk(c,i), Y.chunk(c,idx[i]));
      }
    }

  }
}


#endif // PITTS_MULTIVECTOR_CENTROIDS_HPP
