// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_centroids_impl.hpp
* @brief calculate the weighted sum (centroids) of a set of multi-vectors
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-10
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_CENTROIDS_IMPL_HPP
#define PITTS_MULTIVECTOR_CENTROIDS_IMPL_HPP

// includes
#include <stdexcept>
#include "pitts_multivector_centroids.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement multivector centroids
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


#endif // PITTS_MULTIVECTOR_CENTROIDS_IMPL_HPP
