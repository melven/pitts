// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_from_dense_twosided.hpp
* @brief conversion of a dense tensor to the tensor-train format (based on a hopefully faster TSQR algorithm, two-sided variant)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-08-08
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_HPP
#define PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_HPP

// includes
#include <limits>
#include <vector>
#include <cmath>
#include "pitts_tensortrain.hpp"
#include "pitts_multivector.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate tensor-train decomposition of a tensor stored in fully dense format
  //!
  //! Passing a large enough buffer in work helps to avoid costly reallocations + later page-faults for large data.
  //!
  //! @warning To reduce memory overhead, this function will overwrite the input arguments with temporary data.
  //!          Please pass a copy of the data if you still need it!
  //!
  //! @tparam T         underlying data type (double, complex, ...)
  //!
  //! @param X              input tensor, overwritten and modified output, dimension must be (size/lastDim, lastDim) where lastDim = dimensions.back()
  //! @param dimensions     tensor dimensions, input is interpreted in Fortran storage order (first index changes the fastest)
  //! @param work           buffer for temporary data, will be resized and modified
  //! @param rankTolerance  approximation accuracy (error in the Frobenius norm), used to reduce the TTranks of the resulting tensor train
  //! @param maxRank        maximal TTrank (bond dimension), unbounded by default
  //! @return               resulting tensor train
  //!
  template<typename T>
  TensorTrain<T> fromDense_twoSided(MultiVector<T>& X, MultiVector<T>& work, const std::vector<int>& dimensions, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = -1);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_from_dense_twosided_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_FROM_DENSE_TWOSIDED_HPP
