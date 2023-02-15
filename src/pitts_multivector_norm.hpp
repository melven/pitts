/*! @file pitts_multivector_norm.hpp
* @brief calculate the 2-norm each vector in a multi-vector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_multivector_norm.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_MULTIVECTOR_NORM)
import pitts_multivector_norm;
#define PITTS_MULTIVECTOR_NORM_HPP
#endif

// include guard
#ifndef PITTS_MULTIVECTOR_NORM_HPP
#define PITTS_MULTIVECTOR_NORM_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include <array>
#include <stdexcept>
#ifndef PITTS_USE_MODULES
#include "pitts_eigen.hpp"
#else
#include <string>
#include <complex>
#define EIGEN_CORE_MODULE_H
#include <Eigen/src/Core/util/Macros.h>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/Core/util/ForwardDeclarations.h>
#endif
#include "pitts_multivector.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_multivector_norm;
# define PITTS_MODULE_EXPORT export
#endif

//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
{
  //! calculate the 2-norm of the columns of a multi-vector
  //!
  //! alpha_i <- sqrt(X(:,i)^T X(:,i))
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param X        input multi-vector, dimensions (n, m)
  //! @return         array of norms (m)
  //!
  template<typename T>
  auto norm2(const MultiVector<T>& X)
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

    using arr = Eigen::Array<T, 1, Eigen::Dynamic>;
    arr result = Eigen::Map<arr>(tmp, nCols).sqrt();
    return result;
  }

  // explicit template instantiations
  //template auto norm2<float>(const MultiVector<float>& X);
  //template auto norm2<double>(const MultiVector<double>& X);

}


#endif // PITTS_MULTIVECTOR_NORM_HPP
