/*! @file pitts_tensortrain_operator_impl.hpp
* @brief simple operator in tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2021-02-11
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_IMPL_HPP
#define PITTS_TENSORTRAIN_OPERATOR_IMPL_HPP

// includes
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_normalize.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement TT Op copy
  template<typename T>
  void copy(const TensorTrainOperator<T>& a, TensorTrainOperator<T>& b)
  {
    // check that dimensions match
    if( a.row_dimensions() != b.row_dimensions() )
      throw std::invalid_argument("TensorTrainOperator copy row dimension mismatch!");
    if( a.column_dimensions() != b.column_dimensions() )
      throw std::invalid_argument("TensorTrainOperator copy column dimension mismatch!");

    copy(a.tensorTrain(), b.tensorTrain());
  }


  // implement TT Op axpby
  template<typename T>
  void axpby(T alpha, const TensorTrainOperator<T>& TTOpx, T beta, TensorTrainOperator<T>& TTOpy, T rankTolerance)
  {
    // check that dimensions match
    if( TTOpx.row_dimensions() != TTOpy.row_dimensions() )
      throw std::invalid_argument("TensorTrainOperator axpby row dimension mismatch!");
    if( TTOpx.column_dimensions() != TTOpy.column_dimensions() )
      throw std::invalid_argument("TensorTrainOperator axpby column dimension mismatch!");

    const auto gamma = axpby(alpha, TTOpx.tensorTrain(), beta, TTOpy.tensorTrain(), rankTolerance);
    const int nDim = TTOpy.tensorTrain().dimensions().size();
    if( nDim > 0 )
    {
      Tensor3<T> newSubT;
      copy(TTOpy.tensorTrain().subTensor(nDim-1), newSubT);
      internal::t3_scale(gamma, newSubT);
      TTOpy.tensorTrain().setSubTensor(nDim-1, std::move(newSubT));
    }
  }


  // implement TT Op randomize
  template<typename T>
  void randomize(TensorTrainOperator<T>& TTOp)
  {
    randomize(TTOp.tensorTrain());
  }


  // implement TT Op normalize
  template<typename T>
  T normalize(TensorTrainOperator<T>& TTOp, T rankTolerance, int maxRank)
  {
    return normalize(TTOp.tensorTrain(), rankTolerance, maxRank);
  }
}


#endif // PITTS_TENSORTRAIN_OPERATOR_IMPL_HPP
