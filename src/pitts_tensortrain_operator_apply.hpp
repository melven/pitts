/*! @file pitts_tensortrain_operator_apply.hpp
* @brief apply a tensor train operator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2021-02-12
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_APPLY_HPP
#define PITTS_TENSORTRAIN_OPERATOR_APPLY_HPP

// includes
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_operator.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor3-Operator (e.g. rank-4 tensor) and Tensor3 along middle dimension: A(:,:,*,:) * x(:,*,:)
    template<typename T>
    void apply_contract([[maybe_unused]] const TensorTrainOperator<T>& TTOp, [[maybe_unused]] int iDim, const Tensor3<T>& Aop, const Tensor3<T>& x, Tensor3<T>& y);
  }

  //! Multiply a tensor train operator with a tensor train
  //!
  //! Calculate y <- A * x
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TTOp           tensor train operator
  //! @param TTx            input tensor in tensor train format
  //! @param TTy            output tensor in tensor train format
  //!
  template<typename T>
  void apply(const TensorTrainOperator<T>& TTOp, const TensorTrain<T>& TTx, TensorTrain<T>& TTy);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_operator_apply_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_HPP
