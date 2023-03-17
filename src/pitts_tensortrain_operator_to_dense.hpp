/*! @file pitts_tensortrain_operator_to_dense.hpp
* @brief conversion from the tensor-train operator format into a matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-03-13
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_TO_DENSE_HPP
#define PITTS_TENSORTRAIN_OPERATOR_TO_DENSE_HPP

// includes
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensor2.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! calculate fully dense matrix from a tensor-train operator decomposition
  //!
  //! @tparam T         underlying data type (double, complex, ...)
  //!
  //! @param TTOp       the operator tensor in tensor-train format
  //! @return           operator as dense matrix
  //!
  template<typename T>
  Tensor2<T> toDense(const TensorTrainOperator<T>& TTOp);
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_operator_to_dense_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_OPERATOR_TO_DENSE_HPP
