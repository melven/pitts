// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_scale.hpp
* @brief scale each column in a multi-vector with a scalar
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_SCALE_HPP
#define PITTS_MULTIVECTOR_SCALE_HPP

// includes
#include "pitts_eigen.hpp"
#include "pitts_multivector.hpp"

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
  void scale(const Eigen::ArrayX<T>& alpha, MultiVector<T>& X);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_scale_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_SCALE_HPP
