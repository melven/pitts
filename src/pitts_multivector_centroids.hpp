/*! @file pitts_multivector_centroids.hpp
* @brief calculate the weighted sum (centroids) of a set of multi-vectors
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_multivector_centroids.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_MULTIVECTOR_CENTROIDS)
import pitts_multivector_centroids;
#define PITTS_MULTIVECTOR_CENTROIDS_HPP
#endif

// include guard
#ifndef PITTS_MULTIVECTOR_CENTROIDS_HPP
#define PITTS_MULTIVECTOR_CENTROIDS_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include <vector>
#include <exception>
#include "pitts_multivector.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_multivector_centroids;
# define PITTS_MODULE_EXPORT export
#endif

//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
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

  // explicit template instantiations
  template void centroids<float>(const MultiVector<float>& X, const std::vector<long long>& idx, const std::vector<float>& w, MultiVector<float>& Y);
  template void centroids<double>(const MultiVector<double>& X, const std::vector<long long>& idx, const std::vector<double>& w, MultiVector<double>& Y);
}


#endif // PITTS_MULTIVECTOR_CENTROIDS_HPP
