// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_reverse.hpp
* @brief reverse the order of the dimensions of a tensor train
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-10-20
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_REVERSE_HPP
#define PITTS_TENSORTRAIN_REVERSE_HPP

// includes
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! transpose the rank dimensions of a core: B(j,i,k) = A(k,i,j) (r1 x n x r2 -> r2 x n x r1, no complex conjugation)
    template<typename T>
    void t3_transpose(const Tensor3<T>& A, Tensor3<T>& B);
  }

  //! Reverse the order of the dimensions of a tensor train: y(i_{d-1}, ..., i_0) = x(i_0, ..., i_{d-1})
  //!
  //! The cores of y are the transposed cores of x in reverse order (Y_k = X_{d-1-k}^T with the rank dimensions
  //! swapped), so the TT ranks are reversed as well, including the boundary ranks. A left-orthonormal core of x
  //! becomes a right-orthonormal core of y and vice versa.
  //!
  //! @tparam T     underlying data type (double, complex, ...)
  //! @param TTx    input tensor train
  //! @param TTy    output tensor train, overwritten (also its dimensions)
  //!
  template<typename T>
  void reverse(const TensorTrain<T>& TTx, TensorTrain<T>& TTy);
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_reverse_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_REVERSE_HPP
