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
#include <cmath>
#include "pitts_tensortrain_operator.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"

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
  void applyT(const TensorTrainOperator<T>& TTOp, const TensorTrain<T>& TTx, TensorTrain<T>& TTy)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrainOperator<T>>();

    // check for matching dimensions
    if( TTOp.row_dimensions() != TTx.dimensions() )
      throw std::invalid_argument("TensorTrainOperator: input tensor train dimension mismatch!");
    if( TTOp.column_dimensions() != TTy.dimensions() )
      throw std::invalid_argument("TensorTrainOperator: output tensor train dimension mismatch!");

    // adjust ranks of output tensor train
    const auto newTTranks = internal::dimension_product(TTOp.getTTranks(), TTx.getTTranks());
    TTy.setTTranks(newTTranks);

    // perform actual calculation
    for(int iDim = 0; iDim < TTOp.tensorTrain().subTensors().size(); iDim++)
    {
      const auto& subTOp = TTOp.tensorTrain().subTensors()[iDim];
      const auto& subTx = TTx.subTensors()[iDim];
      auto& subTy = TTy.editableSubTensors()[iDim];

      for(int i1 = 0; i1 < subTOp.r1(); i1++)
        for(int j1 = 0; j1 < subTx.r1(); j1++)
        {
          for(int i2 = 0; i2 < subTOp.r2(); i2++)
            for(int j2 = 0; j2 < subTx.r2(); j2++)
            {
              for(int k = 0; k < subTy.n(); k++)
              {
                T tmp = T(0);
                for(int l = 0; l < subTx.n(); l++)
                  tmp += subTOp(i1,TTOp.index(iDim,l,k),i2) * subTx(j1,l,j2);
                subTy(i1*subTx.r1()+j1,k,i2*subTx.r2()+j2) = tmp;
              }
            }
        }
    }
  }

}


#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_TRANSPOSED_HPP
