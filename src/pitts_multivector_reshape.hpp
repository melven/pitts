// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_reshape.hpp
* @brief adjust the shape of a tall-skinny matrix keeping the data
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-06-29
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_RESHAPE_HPP
#define PITTS_MULTIVECTOR_RESHAPE_HPP

// includes
#include "pitts_multivector.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! reshape a tall-skinny matrix (multivector)
  //!
  //! @tparam T       underlying data type (double, complex, ...)
  //!
  //! @param X        input multi-vector, dimensions (n, m)
  //! @param rows     new number of rows, rows*cols must be equal to n*m
  //! @param cols     new number of columns, rows*cols must be equal to n*m
  //! @param Y        resulting multi-vector, resized to (rows, cols)
  //!
  template<typename T>
  void reshape(const MultiVector<T>& X, long long rows, long long cols, MultiVector<T>& Y);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_reshape_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_RESHAPE_HPP
