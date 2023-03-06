/*! @file pitts_tensortrain_operator_apply_dense.hpp
* @brief apply a tensor train operator to a dense tensor
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-12
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_APPLY_DENSE_HPP
#define PITTS_TENSORTRAIN_OPERATOR_APPLY_DENSE_HPP

// includes
#include "pitts_tensortrain_operator.hpp"
#include "pitts_multivector.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Multiply a tensor train operator with a tensor train
  //!
  //! Calculate y <- A * x
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TTOp           tensor train operator
  //! @param TTx            dense input tensor
  //! @param TTy            dense output tensor
  //!
  template<typename T>
  void apply(const TensorTrainOperator<T>& TTOp, const MultiVector<T>& MVx, MultiVector<T>& MVy);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_operator_apply_dense_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_DENSE_HPP
