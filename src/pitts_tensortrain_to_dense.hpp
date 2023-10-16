// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_to_dense.hpp
* @brief conversion from the tensor-train format into a dense tensor
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-19
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_TO_DENSE_HPP
#define PITTS_TENSORTRAIN_TO_DENSE_HPP

// includes
#include "pitts_tensortrain.hpp"
#include "pitts_multivector.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate fully dense tensor from a tensor-train decomposition (stored as a MultiVector)
  //!
  //! @tparam T         underlying data type (double, complex, ...)
  //!
  //! @param TT         the tensor in tensor-train format
  //! @param X          multivector for storing the tensor in dense format (as one column)
  //!
  template<typename T>
  void toDense(const TensorTrain<T>& TT, MultiVector<T>& X);


  //! calculate fully dense tensor from a tensor-train decomposition
  //!
  //! @tparam T         underlying data type (double, complex, ...)
  //! @tparam Iter      contiguous output iterator to write the dense data
  //!
  //! @param TT             the tensor in tensor-train format
  //! @param first          output iterator that points to the first index, e.g. std::begin(someContainer)
  //! @param last           output iterator that points behind the last index, e.g. std::end(someContainer)
  //!
  template<typename T, class Iter>
  void toDense(const TensorTrain<T>& TT, Iter first, Iter last);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_to_dense_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_TO_DENSE_HPP
