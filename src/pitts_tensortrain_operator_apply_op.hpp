// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_operator_apply_op.hpp
* @brief apply a tensor train operator to another tensor train operator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-29
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_APPLY_OP_HPP
#define PITTS_TENSORTRAIN_OPERATOR_APPLY_OP_HPP

// includes
#include "pitts_tensortrain_operator.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Multiply a tensor train operator with another tensor train operator
  //!
  //! Calculate C <- A * B
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TTOpA           tensor train operator
  //! @param TTOpB           another tensor train operator
  //! @param TTOpC           output tensor train operator
  //!
  template<typename T>
  void apply(const TensorTrainOperator<T>& TTOpA, const TensorTrainOperator<T>& TTOpB, TensorTrainOperator<T>& TTOpC);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_operator_apply_op_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_OP_HPP
