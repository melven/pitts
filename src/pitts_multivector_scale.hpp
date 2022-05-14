/*! @file pitts_multivector_scale.hpp
* @brief scale each column in a multi-vector with a scalar
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_SCALE_HPP
#define PITTS_MULTIVECTOR_SCALE_HPP

// includes
#include <array>
#include <exception>
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options
#include "pitts_multivector.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! scale each column in a multi-vector with a scalar
  //!
  //! X(:,i) <- alpha_i * X(:,i)
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param alpha    array of scaling factors, dimension (m)
  //! @param X        input multi-vector, dimensions (n, m)
  //!
  template<typename T>
  void scale(const Eigen::Array<T,1,Eigen::Dynamic>& alpha, MultiVector<T>& X)
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


#endif // PITTS_MULTIVECTOR_SCALE_HPP
