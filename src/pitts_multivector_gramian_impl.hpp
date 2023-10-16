// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_gramian_impl.hpp
* @brief Calculate the Gram matrix (Gramian) of a multivector (X^T X)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-02-24
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_GRAMIAN_IMPL_HPP
#define PITTS_MULTIVECTOR_GRAMIAN_IMPL_HPP

// includes
#include <memory>
#include "pitts_multivector_gramian.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement multivector gramian
  template<typename T>
  void gramian(const MultiVector<T>& X, Tensor2<T>& G)
  {

    // gather performance data
    const auto n = X.rows();
    const auto m = X.cols();
    const double nd = n, md = m;
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"rows", "cols"},{n,m}}, // arguments
        {{(nd*md*(md+1)/2)*kernel_info::FMA<T>()}, // flops
         {nd*md*kernel_info::Load<T>() + md*md*kernel_info::Store<T>()}} // data transfers
        );

    T tmp[(m*(m+1))/2];
    for(int i = 0; i < (m*(m+1))/2; i++)
      tmp[i] = T(0);

    const auto nChunks = X.rowChunks();

#pragma omp parallel reduction(+:tmp)
    {
      std::unique_ptr<Chunk<T>[]> tmpChunk(new Chunk<T>[m*m]);
      for(int i = 0; i < m; i++)
        for(int j = 0; j <= i; j++)
          tmpChunk[i*m+j] = Chunk<T>{};
#pragma omp for schedule(static)
      for(long iChunk = 0; iChunk < nChunks; iChunk++)
      {
        for(int i = 0; i < m; i++)
          for(int j = 0; j <= i; j++)
            fmadd(X.chunk(iChunk,i),X.chunk(iChunk,j), tmpChunk[i*m+j]);
      }

      int k = 0;
      for(int i = 0; i < m; i++)
        for(int j = 0; j <= i; j++)
          tmp[k++] = sum(tmpChunk[i*m+j]);
    }

    G.resize(m,m);
    {
      int k = 0;
      for(int i = 0; i < m; i++)
        for(int j = 0; j <= i; j++)
          G(i,j) = G(j,i) = tmp[k++];
    }
  }

}


#endif // PITTS_MULTIVECTOR_GRAMIAN_IMPL_HPP
