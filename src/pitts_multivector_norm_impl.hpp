/*! @file pitts_multivector_norm_impl.hpp
* @brief calculate the 2-norm each vector in a multi-vector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_NORM_IMPL_HPP
#define PITTS_MULTIVECTOR_NORM_IMPL_HPP

// includes
#include <array>
#include <stdexcept>
#include "pitts_multivector_norm.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement multivector norm
  template<typename T>
  Eigen::ArrayX<T> norm2(const MultiVector<T>& X)
  {
    const auto nChunks = X.rowChunks();
    const auto nCols = X.cols();

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"nChunks", "nCols"},{nChunks, nCols}}, // arguments
        {{nCols*nChunks*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {nChunks*nCols*kernel_info::Load<Chunk<T>>() + nCols*kernel_info::Store<T>()}} // data transfers
        );

    T tmp[nCols];
    for(int iCol = 0; iCol < nCols; iCol++)
      tmp[iCol] = T(0);
#pragma omp parallel reduction(+:tmp)
    {
      for(int iCol = 0; iCol < nCols; iCol++)
      {
        Chunk<T> tmpChunk{};
#pragma omp for schedule(static) nowait
        for(int iChunk = 0; iChunk < nChunks; iChunk++)
          fmadd(X.chunk(iChunk,iCol), X.chunk(iChunk,iCol), tmpChunk);
        tmp[iCol] = sum(tmpChunk);
      }
    }

    using arr = Eigen::ArrayX<T>;
    arr result = Eigen::Map<arr>(tmp, nCols).sqrt();
    return result;
  }

}


#endif // PITTS_MULTIVECTOR_NORM_IMPL_HPP