/*! @file pitts_multivector_scale.hpp
* @brief scale each column in a multi-vector with a scalar
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_multivector_scale.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_MULTIVECTOR_SCALE)
import pitts_multivector_scale;
#define PITTS_MULTIVECTOR_SCALE_HPP
#endif

// include guard
#ifndef PITTS_MULTIVECTOR_SCALE_HPP
#define PITTS_MULTIVECTOR_SCALE_HPP

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
//#define EIGEN_CORE_MODULE_H
//#include <Eigen/src/Core/util/Macros.h>
//#include <Eigen/src/Core/util/Constants.h>
//#include <Eigen/src/Core/util/ForwardDeclarations.h>
namespace Eigen
{
#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION Eigen::RowMajor
#else
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION Eigen::ColMajor
#endif
  enum StorageOptions {
    ColMajor = 0,
    RowMajor = 0x1,  // it is only a coincidence that this is equal to RowMajorBit -- don't rely on that
    AutoAlign = 0,
    DontAlign = 0x2
  };
  const int Dynamic = -1;
  template<typename Scalar_, int Rows_, int Cols_,
           int Options_ = AutoAlign |
                            ( (Rows_==1 && Cols_!=1) ? Eigen::RowMajor
                            : (Cols_==1 && Rows_!=1) ? Eigen::ColMajor
                            : EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION ),
           int MaxRows_ = Rows_, int MaxCols_ = Cols_> class Array;
}
#endif
#include "pitts_multivector.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_multivector_scale;
# define PITTS_MODULE_EXPORT export
#endif

//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
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

  // explicit template instantiations
  //template void scale<float>(const Eigen::Array<float,1,Eigen::Dynamic>& alpha, MultiVector<float>& X);
  //template void scale<double>(const Eigen::Array<double,1,Eigen::Dynamic>& alpha, MultiVector<double>& X);

}


#endif // PITTS_MULTIVECTOR_SCALE_HPP
