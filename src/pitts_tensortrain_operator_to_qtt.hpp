// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_operator_to_qtt.hpp
* @brief converts a tensor-train operator to QTT format (splitting all dimensions to (2x2)^d)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-06-10
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_TO_QTT_HPP
#define PITTS_TENSORTRAIN_OPERATOR_TO_QTT_HPP

// includes
#include "pitts_tensortrain_operator.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate quantized tensor-train (QTT) format from a given tensor-train operator
  //!
  //! @tparam T         underlying data type (double, complex, ...)
  //!
  //! @param TTOp       the operator tensor in tensor-train format, each dimension must be a power of two (for now!)
  //! @return           operator in QTT format ((2x2)^d)
  //!
  template<typename T>
  TensorTrainOperator<T> toQtt(const TensorTrainOperator<T>& TTOp);
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_operator_to_qtt_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_OPERATOR_TO_QTT_HPP
