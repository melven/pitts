/*! @file pitts_multivector_axpby.hpp
* @brief calculate the scaled addition of the pair-wise columns of two multi-vectors
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_multivector_axpby.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_MULTIVECTOR_AXPBY)
import pitts_multivector_axpby;
#define PITTS_MULTIVECTOR_AXPBY_HPP
#endif

// include guard
#ifndef PITTS_MULTIVECTOR_AXPBY_HPP
#define PITTS_MULTIVECTOR_AXPBY_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include <array>
#include <exception>
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
export module pitts_multivector_axpby;
# define PITTS_MODULE_EXPORT export
#endif

//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
{
  //! calculate the pair-wise scaled addition of the columns of two multi-vectors
  //!
  //! Calculates Y(:,i) <- alpha_i*X(:,i) + Y(:,i)
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param alpha    array of scaling factors, dimension (m)
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param Y        resulting multi-vector, dimensions (n, m)
  //!
  template<typename T>
  void axpy(const Eigen::Array<T, 1, Eigen::Dynamic>& alpha, const MultiVector<T>& X, MultiVector<T>& Y)
  {
    // check dimensions
    if( X.cols() != Y.cols() || alpha.size() != X.cols() )
      throw std::invalid_argument("MultiVector::axpy: cols dimension mismatch!");

    if( X.rows() != Y.rows() )
      throw std::invalid_argument("MultiVector::axpy: rows dimension mismatch!");

    const auto nChunks = X.rowChunks();
    const auto nCols = Y.cols();

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"nChunks", "nCols"},{nChunks, nCols}}, // arguments
        {{nChunks*nCols*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {nChunks*nCols*kernel_info::Load<Chunk<T>>() + nChunks*nCols*kernel_info::Update<T>()}} // data transfers
        );

#pragma omp parallel
    {
      for(int iCol = 0; iCol < nCols; iCol++)
      {
#pragma omp for schedule(static) nowait
        for(int iChunk = 0; iChunk < nChunks; iChunk++)
          fmadd(alpha(iCol), X.chunk(iChunk,iCol), Y.chunk(iChunk,iCol));
      }
    }
  }

  //! calculate the pair-wise scaled addition of the columns of two multi-vectors, and the norm of the result
  //!
  //! Calculates Y(:,i) <- alpha_i*X(:,i) + Y(:,i)
  //!            gamma_i <- ||Y(:,i)||_2
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param alpha    array of scaling factors, dimension (m)
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param Y        resulting multi-vector, dimensions (n, m)
  //! @returns        array of norms of the resulting y, dimension (m)
  //!
  template<typename T>
  auto axpy_norm2(const Eigen::Array<T, 1, Eigen::Dynamic>& alpha, const MultiVector<T>& X, MultiVector<T>& Y)
  {
    // check dimensions
    if( X.cols() != Y.cols() || alpha.size() != X.cols() )
      throw std::invalid_argument("MultiVector::axpy_norm2: cols dimension mismatch!");

    if( X.rows() != Y.rows() )
      throw std::invalid_argument("MultiVector::axpy_norm2: rows dimension mismatch!");

    const auto nChunks = X.rowChunks();
    const auto nCols = Y.cols();

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"nChunks", "nCols"},{nChunks, nCols}}, // arguments
        {{2*nChunks*nCols*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {nChunks*nCols*kernel_info::Load<Chunk<T>>() + nChunks*nCols*kernel_info::Update<T>() + nCols*kernel_info::Store<T>()}} // data transfers
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
        {
          fmadd(alpha(iCol), X.chunk(iChunk,iCol), Y.chunk(iChunk,iCol));
          fmadd(Y.chunk(iChunk,iCol), Y.chunk(iChunk,iCol), tmpChunk);
        }
        tmp[iCol] = sum(tmpChunk);
      }
    }

    using arr = Eigen::Array<T, 1, Eigen::Dynamic>;
    arr result = Eigen::Map<arr>(tmp, nCols).sqrt();
    return result;
  }

  //! calculate the pair-wise scaled addition of the columns of two multi-vectors, and the dot product of the result with the columsn of a third multi-vector
  //!
  //! Calculates Y(:,i) <- alpha_i*X(:,i) + Y(:,i)
  //!            gamma_i <- Y(:,i)^T * Z(:,i)
  //!            
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param alpha    array of scaling factors, dimension (m)
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param Y        resulting multi-vector, dimensions (n, m)
  //! @param Z        third multi-vector, dimensions (n,m)
  //! @returns        array of dot products between Y and Z, dimension (m)
  //!
  template<typename T>
  auto axpy_dot(const Eigen::Array<T, 1, Eigen::Dynamic>& alpha, const MultiVector<T>& X, MultiVector<T>& Y, const MultiVector<T>& Z)
  {
    // check dimensions
    if( X.cols() != Y.cols() || X.cols() != Z.cols() || alpha.size() != X.cols() )
      throw std::invalid_argument("MultiVector::axpy_dot: cols dimension mismatch!");

    if( X.rows() != Y.rows() || X.rows() != Z.rows() )
      throw std::invalid_argument("MultiVector::axpy_dot: rows dimension mismatch!");

    const auto nChunks = X.rowChunks();
    const auto nCols = Y.cols();

    // gather performance data
    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"nChunks", "nCols"},{nChunks, nCols}}, // arguments
        {{2*nChunks*nCols*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {nChunks*nCols*kernel_info::Load<Chunk<T>>() + nChunks*nCols*kernel_info::Update<T>() + nCols*kernel_info::Store<T>()}} // data transfers
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
        {
          fmadd(alpha(iCol), X.chunk(iChunk,iCol), Y.chunk(iChunk,iCol));
          fmadd(Y.chunk(iChunk,iCol), Z.chunk(iChunk,iCol), tmpChunk);
        }
        tmp[iCol] = sum(tmpChunk);
      }
    }

    using arr = Eigen::Array<T, 1, Eigen::Dynamic>;
    arr result = Eigen::Map<arr>(tmp, nCols);
    return result;
  }

  // explicit template instantiations
  //template void axpy<float>(const Eigen::Array<float, 1, Eigen::Dynamic>& alpha, const MultiVector<float>& X, MultiVector<float>& Y);
  //template auto axpy_norm2<float>(const Eigen::Array<float, 1, Eigen::Dynamic>& alpha, const MultiVector<float>& X, MultiVector<float>& Y);
  //template auto axpy_dot<float>(const Eigen::Array<float, 1, Eigen::Dynamic>& alpha, const MultiVector<float>& X, MultiVector<float>& Y, const MultiVector<float>& Z);
  //template void axpy<double>(const Eigen::Array<double, 1, Eigen::Dynamic>& alpha, const MultiVector<double>& X, MultiVector<double>& Y);
  //template auto axpy_norm2<double>(const Eigen::Array<double, 1, Eigen::Dynamic>& alpha, const MultiVector<double>& X, MultiVector<double>& Y);
  //template auto axpy_dot<double>(const Eigen::Array<double, 1, Eigen::Dynamic>& alpha, const MultiVector<double>& X, MultiVector<double>& Y, const MultiVector<double>& Z);
}


#endif // PITTS_MULTIVECTOR_AXPBY_HPP
