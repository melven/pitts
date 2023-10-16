// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_scale_impl.hpp
* @brief scale each column in a multi-vector with a scalar
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_SCALE_IMPL_HPP
#define PITTS_MULTIVECTOR_SCALE_IMPL_HPP

// includes
#include <stdexcept>
#include "pitts_multivector_scale.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement multivector scale
  template<typename T>
  void scale(const Eigen::ArrayX<T>& alpha, MultiVector<T>& X)
  {
    const auto nChunks = X.rowChunks();
    const auto nCols = X.cols();

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"nChunks", "nCols"},{nChunks, nCols}}, // arguments
        {{nCols*nChunks*Chunk<T>::size*kernel_info::Mult<T>()}, // flops
         {nCols*kernel_info::Load<T>() + nChunks*nCols*kernel_info::Update<Chunk<T>>()}} // data transfers
        );

#pragma omp parallel
    {
      for(int iCol = 0; iCol < nCols; iCol++)
      {
#pragma omp for schedule(static) nowait
        for(int iChunk = 0; iChunk < nChunks; iChunk++)
          mul(alpha(iCol), X.chunk(iChunk,iCol), X.chunk(iChunk,iCol));
      }
    }
  }

}


#endif // PITTS_MULTIVECTOR_SCALE_IMPL_HPP
