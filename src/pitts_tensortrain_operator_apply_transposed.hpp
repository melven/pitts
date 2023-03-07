/*! @file pitts_tensortrain_operator_apply_transposed.hpp
* @brief apply the tranposed of a tensor train operator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-27
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_APPLY_TRANSPOSED_HPP
#define PITTS_TENSORTRAIN_OPERATOR_APPLY_TRANSPOSED_HPP

// includes
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_operator.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Multiply a tensor train operator with a tensor train
  //!
  //! Calculate y <- A^T * x
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TTOp           tensor train operator
  //! @param TTx            input tensor in tensor train format
  //! @param TTy            output tensor in tensor train format
  //!
  template<typename T>
  void applyT(const TensorTrainOperator<T>& TTOp, const TensorTrain<T>& TTx, TensorTrain<T>& TTy);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_operator_apply_transposed_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_TRANSPOSED_HPP
