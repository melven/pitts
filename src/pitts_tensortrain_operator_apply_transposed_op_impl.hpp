/*! @file pitts_tensortrain_operator_apply_transposed_op_impl.hpp
* @brief apply the transposed of a tensor train operator to another tensor train operator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-29
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_APPLY_TRANSPOSED_OP_IMPL_HPP
#define PITTS_TENSORTRAIN_OPERATOR_APPLY_TRANSPOSED_OP_IMPL_HPP

// includes
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <vector>
#include "pitts_tensortrain_operator_apply_transposed_op.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_performance.hpp"
#include "pitts_tensor3.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor3-Operator (e.g. rank-4 tensor) and another Tensor3-Operator along middle dimension: A(:,*,:,:) * B(:,*,:,:)
    template<typename T>
    void applyT_contract_op(const TensorTrainOperator<T>& TTOpA,
                           const TensorTrainOperator<T>& TTOpB,
                           [[maybe_unused]] const TensorTrainOperator<T>& TTOpC,
                           int iDim,
                           const Tensor3<T>& Aop,
                           const Tensor3<T>& Bop,
                           Tensor3<T>& Cop)
    {
      const auto rA1 = Aop.r1();
      const auto rA2 = Aop.r2();
      const auto rB1 = Bop.r1();
      const auto rB2 = Bop.r2();
      const auto mA = TTOpA.column_dimensions()[iDim];
      const auto nB = TTOpB.row_dimensions()[iDim];
      const auto mB = TTOpB.column_dimensions()[iDim];

      Cop.resize(rA1*rB1, mA*mB, rA2*rB2);

      const auto indexA = [nB](int k, int l) {return k + nB*l;};
      const auto indexB = [nB](int k, int l) {return k + nB*l;};
      const auto indexC = [mA](int k, int l) {return k + mA*l;};


      // check that the index function is ok...
      for(int k = 0; k < mA; k++)
        for(int l = 0; l < nB; l++)
        {
          assert(indexA(k,l) == TTOpA.index(iDim, k, l));
        }
      for(int k = 0; k < nB; k++)
        for(int l = 0; l < mB; l++)
        {
          assert(indexB(k,l) == TTOpB.index(iDim, k, l));
        }
      for(int k = 0; k < mA; k++)
        for(int l = 0; l < mB; l++)
        {
          assert(indexC(k,l) == TTOpC.index(iDim, k, l));
        }

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"rA1", "mA", "nB", "mB", "rA2", "rB1", "rB2"},{rA1, mA, nB, mB, rA2, rB1, rB2}}, // arguments
        {{rA1*rA2*mA*nB*mB*rB1*rB2*kernel_info::FMA<T>()}, // flops
         {(rA1*mA*nB*rA2+rB1*nB*mB*rB2)*kernel_info::Load<Chunk<T>>() + (rA1*rB1*mA*mB*rA2*rB2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

#pragma omp parallel for collapse(6) schedule(static)
      for(int i1 = 0; i1 < rA1; i1++)
        for(int j1 = 0; j1 < rB1; j1++)
        {
          for(int i2 = 0; i2 < rA2; i2++)
            for(int j2 = 0; j2 < rB2; j2++)
            {
              for(int k1 = 0; k1 < mA; k1++)
                for(int k2 = 0; k2 < mB; k2++)
                {
                  T tmp = T(0);
                  for(int l = 0; l < nB; l++)
                    tmp += Aop(i1,indexA(l,k1),i2) * Bop(j1,indexB(l,k2),j2);
                  Cop(i1*rB1+j1,indexC(k1,k2),i2*rB2+j2) = tmp;
                }
            }
        }
    }
  }


  // implement TT Op apply transposed
  template<typename T>
  void applyT(const TensorTrainOperator<T>& TTOpA, const TensorTrainOperator<T>& TTOpB, TensorTrainOperator<T>& TTOpC)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrainOperator<T>>();

    // check for matching dimensions
    if( TTOpA.row_dimensions() != TTOpB.row_dimensions() )
      throw std::invalid_argument("TensorTrainOperator: input tensor train operator dimension mismatch!");
    if( TTOpA.column_dimensions() != TTOpC.row_dimensions() )
      throw std::invalid_argument("TensorTrainOperator: output tensor train operator dimension mismatch!");
    if( TTOpB.column_dimensions() != TTOpC.column_dimensions() )
      throw std::invalid_argument("TensorTrainOperator: input/output tensor train operator dimensions mismatch!");

    // perform actual calculation
    const int nDim = TTOpA.tensorTrain().dimensions().size();
    std::vector<Tensor3<T>> subTOpC(nDim);
    for(int iDim = 0; iDim < nDim; iDim++)
    {
      const auto& subTOpA = TTOpA.tensorTrain().subTensor(iDim);
      const auto& subTOpB = TTOpB.tensorTrain().subTensor(iDim);

      internal::applyT_contract_op(TTOpA, TTOpB, TTOpC, iDim, subTOpA, subTOpB, subTOpC[iDim]);
    }
    TTOpC.tensorTrain().setSubTensors(0, std::move(subTOpC));
  }

}


#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_TRANSPOSED_OP_IMPL_HPP
