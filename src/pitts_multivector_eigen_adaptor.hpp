// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_eigen_adaptor.hpp
* @brief Get an Eigen::Map for a simple PITTS::MultiVector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-31
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_EIGEN_ADAPTOR_HPP
#define PITTS_MULTIVECTOR_EIGEN_ADAPTOR_HPP

// includes
#include "pitts_multivector.hpp"
#include "pitts_eigen.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Get a const Eigen::Map for a MultiVector
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  auto ConstEigenMap(const MultiVector<T>& mv)
  {
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    return Eigen::Map<const Matrix, EigenAligned, Eigen::OuterStride<> >(&mv(0,0), mv.rows(), mv.cols(), Eigen::OuterStride<>(mv.colStrideChunks()*Chunk<T>::size));
  }

  //! Get a mutable Eigen::Map for a MultiVector
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  auto EigenMap(MultiVector<T>& mv)
  {
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    return Eigen::Map<Matrix, EigenAligned, Eigen::OuterStride<> >(&mv(0,0), mv.rows(), mv.cols(), Eigen::OuterStride<>(mv.colStrideChunks()*Chunk<T>::size));
  }
}


#endif // PITTS_MULTIVECTOR_EIGEN_ADAPTOR_HPP
