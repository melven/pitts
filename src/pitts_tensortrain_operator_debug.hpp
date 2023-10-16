// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_operator_debug.hpp
* @brief Debug helper functions for PITT::TensorTrainOperator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-01-14
*
* Currently provides operator overloads to calculate with tensor-train operators.
* As this generates temporaries and uses sub-optimal kernels, this is only intended for e.g. assert-statements.
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_DEBUG_HPP
#define PITTS_TENSORTRAIN_OPERATOR_DEBUG_HPP

// includes
#include <vector>
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_operator_apply_op.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain_normalize.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for debugging functionality
  namespace debug
  {
    //! operator overload for TT scaling (alpha*TT)
    template<typename T>
    [[nodiscard]] TensorTrainOperator<T> operator*(T alpha, const TensorTrainOperator<T>& ttOp)
    {
        TensorTrainOperator<T> result = ttOp;
        if( ttOp.row_dimensions().size() > 0 )
        {
            PITTS::Tensor3<T> newSubT0;
            copy(result.tensorTrain().subTensor(0), newSubT0);
            internal::t3_scale(alpha, newSubT0);
            result.tensorTrain().setSubTensor(0, std::move(newSubT0));
        }
        return result;
    }

    //! operator overload for TT scaling (TT*alpha)
    template<typename T>
    [[nodiscard]] TensorTrainOperator<T> operator*(const TensorTrainOperator<T>& ttOp, T alpha) {return alpha*ttOp;}

    //! operator overload for TT addition (axpy with copy)
    template<typename T>
    [[nodiscard]] TensorTrainOperator<T> operator+(const TensorTrainOperator<T>& ttOpA, const TensorTrainOperator<T>& ttOpB)
    {
        TensorTrainOperator<T> result = ttOpB;
        axpby(T(1), ttOpA, T(1), result, T(0));
        return result;
    }

    //! operator overload for TT subtraction (axpy with copy)
    template<typename T>
    [[nodiscard]] TensorTrainOperator<T> operator-(const TensorTrainOperator<T>& ttOpA, const TensorTrainOperator<T>& ttOpB)
    {
        TensorTrainOperator<T> result = ttOpB;
        axpby(T(1), ttOpA, T(-1), result, T(0));
        return result;
    }

    //! operator overload for TT scaling (TT/alpha)
    template<typename T>
    [[nodiscard]] TensorTrainOperator<T> operator/(const TensorTrainOperator<T>& ttOp, T alpha) {return T(1)/alpha * ttOp;}

    //! operator overload for ttOp * ttX
    template<typename T>
    [[nodiscard]] TensorTrain<T> operator*(const TensorTrainOperator<T>& ttOp, const TensorTrain<T>& ttX)
    {
      // workaround for boundary rank
      TensorTrain<T> ttY( ttOp.row_dimensions() == ttOp.column_dimensions() ? ttX.dimensions() : ttOp.row_dimensions() );
      apply(ttOp, ttX, ttY);
      return ttY;
    }

    //! operator overload for ttOpA * ttOpB
    template<typename T>
    [[nodiscard]] TensorTrainOperator<T> operator*(const TensorTrainOperator<T>& ttOpA, const TensorTrainOperator<T>& ttOpB)
    {
      TensorTrainOperator<T> ttOpC(ttOpA.row_dimensions(), ttOpB.column_dimensions());
      apply(ttOpA, ttOpB, ttOpC);
      return ttOpC;
    }

    //! transpose tensor-train operator
    template<typename T>
    [[nodiscard]] TensorTrainOperator<T> transpose(const TensorTrainOperator<T>& ttOp)
    {
      TensorTrainOperator<T> result(ttOp.column_dimensions(), ttOp.row_dimensions());
      result.setTTranks(ttOp.getTTranks());

      for(int iDim = 0; iDim < ttOp.row_dimensions().size(); iDim++)
      {
        int n = ttOp.row_dimensions()[iDim];
        int m = ttOp.column_dimensions()[iDim];
        const auto& subT = ttOp.tensorTrain().subTensor(iDim);
        Tensor3<T> newSubT(subT.r1(), subT.n(), subT.r2());
        for(int i = 0; i < subT.r1(); i++)
          for(int j = 0; j < n; j++)
            for(int k = 0; k < m; k++)
              for(int l = 0; l < subT.r2(); l++)
                newSubT(i, result.index(iDim, k, j), l) = subT(i, ttOp.index(iDim, j, k), l);
        result.tensorTrain().setSubTensor(iDim, std::move(newSubT));
      }
      return result;
    }
  }
}

#endif // PITTS_TENSORTRAIN_OPERATOR_DEBUG_HPP
