// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor3_combine.hpp
* @brief contract two simple rank-3 tensors (along third and first dimension)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-22
*
**/

// include guard
#ifndef PITTS_TENSOR3_COMBINE_HPP
#define PITTS_TENSOR3_COMBINE_HPP

// includes
#include "pitts_tensor3.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate the result of multiplying two rank-3 tensors (contraction of third and first dimension)
  //!
  //! The resulting tensor is t3c with
  //!   t3c_(i,k,j) = sum_l t3a_(i,k1,l) * t3b_(l,k2,j)
  //! with a tensor product of the second dimensions:
  //! * for swap=false, we use k=k2*n1+k1
  //! * for swap=true,  we use k=k2+n2*k1
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param t3a    first rank-3 tensor
  //! @param t3b    second rank-3 tensor
  //! @param swap   store second dimension in "transposed" order
  //! @return       resulting t3c (see formula above)
  //!
  template<typename T>
  Tensor3<T> combine(const Tensor3<T>& t3a, const Tensor3<T>& t3b, bool swap = false);
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensor3_combine_impl.hpp"
#endif

#endif // PITTS_TENSOR3_COMBINE_HPP
