/*! @file pitts_multivector_impl.hpp
* @brief Rank-2 tensor that represents a set of large vectors (e.g. leading dimension is large)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-09
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_IMPL_HPP
#define PITTS_MULTIVECTOR_IMPL_HPP

// includes
#include "pitts_multivector.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement multivector copy
  template<typename T>
  void copy(const MultiVector<T>& a, MultiVector<T>& b)
  {
    const auto rows = a.rows();
    const auto rowChunks = a.rowChunks();
    const auto cols = a.cols();

    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"ros", "cols"}, {rows, cols}},   // arguments
        {{double(rows)*cols*kernel_info::NoOp<T>()},    // flops
         {double(rows)*cols*kernel_info::Store<T>() + double(rows)*cols*kernel_info::Load<T>()}}  // data
        );

    b.resize(rows, cols);

#pragma omp parallel
    {
      for(long long j = 0; j < cols; j++)
      {
#pragma omp for schedule(static) nowait
        for(long long iChunk = 0; iChunk < rowChunks; iChunk++)
          streaming_store(a.chunk(iChunk,j), b.chunk(iChunk,j));
      }
    }
  }
}


#endif // PITTS_MULTIVECTOR_IMPL_HPP
