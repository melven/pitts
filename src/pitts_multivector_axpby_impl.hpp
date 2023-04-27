/*! @file pitts_multivector_axpby_impl.hpp
* @brief calculate the scaled addition of the pair-wise columns of two multi-vectors
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_AXPBY_IMPL_HPP
#define PITTS_MULTIVECTOR_AXPBY_IMPL_HPP

// includes
#include <array>
#include <stdexcept>
#include "pitts_eigen.hpp"
#include "pitts_multivector_axpby.hpp"
#include "pitts_performance.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement MultiVector axpy
  template<typename T>
  void axpy(const Eigen::ArrayX<T>& alpha, const MultiVector<T>& X, MultiVector<T>& Y)
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

#pragma omp parallel if(nChunks > 50)
    {
      for(int iCol = 0; iCol < nCols; iCol++)
      {
#pragma omp for schedule(static) nowait
        for(int iChunk = 0; iChunk < nChunks; iChunk++)
          fmadd(alpha(iCol), X.chunk(iChunk,iCol), Y.chunk(iChunk,iCol));
      }
    }
  }

  // implement MultiVector axpy+norm
  template<typename T>
  Eigen::ArrayX<T> axpy_norm2(const Eigen::ArrayX<T>& alpha, const MultiVector<T>& X, MultiVector<T>& Y)
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
#pragma omp parallel reduction(+:tmp) if(nChunks > 50)
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

    using arr = Eigen::ArrayX<T>;
    arr result = Eigen::Map<arr>(tmp, nCols).sqrt();
    return result;
  }

  // implement MultiVector axpy+dot
  template<typename T>
  Eigen::ArrayX<T> axpy_dot(const Eigen::ArrayX<T>& alpha, const MultiVector<T>& X, MultiVector<T>& Y, const MultiVector<T>& Z)
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
         {nChunks*nCols*kernel_info::Load<Chunk<T>>() + nChunks*nCols*kernel_info::Update<Chunk<T>>() + nCols*kernel_info::Store<T>()}} // data transfers
        );

    T tmp[nCols];
    for(int iCol = 0; iCol < nCols; iCol++)
      tmp[iCol] = T(0);
#pragma omp parallel reduction(+:tmp) if(nChunks > 50)
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

    using arr = Eigen::ArrayX<T>;
    arr result = Eigen::Map<arr>(tmp, nCols);
    return result;
  }
}


#endif // PITTS_MULTIVECTOR_AXPBY_IMPL_HPP
