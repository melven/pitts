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
#include <vector>
#include <cassert>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor3-Operator (e.g. rank-4 tensor) and Tensor3 along middle dimension: A(:,*,:,:) * x(:,*,:)
    template<typename T>
    void applyT_contract([[maybe_unused]] const TensorTrainOperator<T>& TTOp, [[maybe_unused]] int iDim, const Tensor3<T>& Aop, const Tensor3<T>& x, Tensor3<T>& y)
    {
      const auto rA1 = Aop.r1();
      const auto rA2 = Aop.r2();
      const auto r1 = x.r1();
      const auto r2 = x.r2();
      const auto n = x.n();
      const auto m = Aop.n() / n;

      y.resize(rA1*r1, m, rA2*r2);

      const auto index = [n](int k, int l)
      {
        return k + n*l;
      };

      // check that the index function is ok...
      for(int k = 0; k < n; k++)
        for(int l = 0; l < m; l++)
        {
          assert(index(k,l) == TTOp.index(iDim, k, l));
        }

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"rA1", "n", "m", "rA2", "r1", "r2"},{rA1, n, m, rA2, r1, r2}}, // arguments
        {{rA1*rA2*n*m*r1*r2*kernel_info::FMA<T>()}, // flops
         {(rA1*m*n*rA2+r1*n*r2)*kernel_info::Load<Chunk<T>>() + (rA1*r1*m*rA2*r2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      for(int i1 = 0; i1 < rA1; i1++)
        for(int j1 = 0; j1 < r1; j1++)
        {
          for(int i2 = 0; i2 < rA2; i2++)
            for(int j2 = 0; j2 < r2; j2++)
            {
              for(int k = 0; k < m; k++)
              {
                T tmp = T(0);
                for(int l = 0; l < n; l++)
                  tmp += Aop(i1,index(l,k),i2) * x(j1,l,j2);
                y(i1*r1+j1,k,i2*r2+j2) = tmp;
              }
            }
        }
    }
  }

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

    // perform actual calculation
    const int nDim = TTOp.tensorTrain().dimensions().size();
    std::vector<Tensor3<T>> subTy(nDim);
    for(int iDim = 0; iDim < nDim; iDim++)
    {
      const auto& subTOp = TTOp.tensorTrain().subTensor(iDim);
      const auto& subTx = TTx.subTensor(iDim);

      internal::applyT_contract(TTOp, iDim, subTOp, subTx, subTy[iDim]);
    }
    TTy.setSubTensors(0, std::move(subTy));
  }

}


#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_TRANSPOSED_HPP
