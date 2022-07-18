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
#include <cmath>
#include "pitts_tensortrain_operator.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor3-Operator (e.g. rank-4 tensor) and Tensor3 along middle dimension: A(:,:,*,:) * x(:,*,:)
    template<typename T>
    void apply_contract(const TensorTrainOperator<T>& TTOp, int iDim, const Tensor3<T>& Aop, const Tensor3<T>& x, Tensor3<T>& y)
    {
      const auto rA1 = Aop.r1();
      const auto rA2 = Aop.r2();
      const auto r1 = x.r1();
      const auto r2 = x.r2();
      const auto m = x.n();
      const auto n = Aop.n() / m;

      y.resize(rA1*r1, n, rA2*r2);

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
         {(rA1*m*n*rA2+r1*m*r2)*kernel_info::Load<Chunk<T>>() + (rA1*r1*n*rA2*r2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      for(int i1 = 0; i1 < rA1; i1++)
        for(int j1 = 0; j1 < r1; j1++)
        {
          for(int i2 = 0; i2 < rA2; i2++)
            for(int j2 = 0; j2 < r2; j2++)
            {
              for(int k = 0; k < n; k++)
              {
                T tmp = T(0);
                for(int l = 0; l < m; l++)
                  tmp += Aop(i1,index(k,l),i2) * x(j1,l,j2);
                y(i1*r1+j1,k,i2*r2+j2) = tmp;
              }
            }
        }
    }

    //! contract Tensor3-Operator (e.g. rank-4 tensor) and Tensor3 along some dimensions: A(0,:,*,*) * x(*,:,:)
    template<typename T>
    void apply_contract_leftBoundaryRank(const TensorTrainOperator<T>& TTOp, int iDim, const Tensor3<T>& Aop, const Tensor3<T>& x, Tensor3<T>& y)
    {
      assert(Aop.r1() == 1);
      const auto rA2 = Aop.r2();
      const auto r1 = x.r1();
      const auto xn = x.n();
      const auto r2 = x.r2();
      const auto m = r1 / rA2;
      const auto n = Aop.n() / m;
      assert(r1 == rA2*m);
      assert(TTOp.row_dimensions()[iDim] == n);
      assert(TTOp.column_dimensions()[iDim] == m);

      y.resize(n, xn, r2);

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
        {{"rA2", "r1", "xn", "r2", "n", "m"},{rA2, r1, xn, r2, n, m}}, // arguments
        {{m*r1*xn*r2*kernel_info::FMA<T>()}, // flops
         {(m*n*rA2+r1*xn*r2)*kernel_info::Load<Chunk<T>>() + (m*xn*r2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      for(int i1 = 0; i1 < n; i1++)
        for(int k = 0; k < xn; k++)
          for(int j1 = 0; j1 < r2; j1++)
          {
            T tmp(0);
            for(int l = 0; l < m; l++)
              for(int j2 = 0; j2 < rA2; j2++)
                tmp += Aop(0,index(i1,l),j2) * x(l+j2*m,k,j1);
            y(i1,k,j1) = tmp;
          }
    }

    //! contract Tensor3-Operator (e.g. rank-4 tensor) and Tensor3 along some dimensions: x(:,:,*) * A(*,:,*,0)
    template<typename T>
    void apply_contract_rightBoundaryRank(const Tensor3<T>& x, const TensorTrainOperator<T>& TTOp, int iDim, const Tensor3<T>& Aop, Tensor3<T>& y)
    {
      const auto rA1 = Aop.r1();
      assert(Aop.r2() == 1);
      const auto r1 = x.r1();
      const auto xn = x.n();
      const auto r2 = x.r2();
      const auto m = r2 / rA1;
      const auto n = Aop.n() / m;
      assert(r2 == rA1*m);
      assert(TTOp.row_dimensions()[iDim] == n);
      assert(TTOp.column_dimensions()[iDim] == m);

      y.resize(r1, xn, n);

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
        {{"rA1", "r1", "xn", "r2", "n", "m"},{rA1, r1, xn, r2, n, m}}, // arguments
        {{r1*xn*r2*m*kernel_info::FMA<T>()}, // flops
         {(rA1*n*m+r1*xn*r2)*kernel_info::Load<Chunk<T>>() + (r1*xn*m)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      for(int i1 = 0; i1 < r1; i1++)
        for(int k = 0; k < xn; k++)
          for(int j1 = 0; j1 < n; j1++)
          {
            T tmp(0);
            for(int l = 0; l < m; l++)
              for(int i2 = 0; i2 < rA1; i2++)
                tmp += Aop(i2,index(j1,l),0) * x(i1,k,l+i2*m);
            y(i1,k,j1) = tmp;
          }
    }
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
  void apply(const TensorTrainOperator<T>& TTOp, const TensorTrain<T>& TTx, TensorTrain<T>& TTy)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrainOperator<T>>();

    // check for identical number of dimensions (to handle special case with boundary ranks)
    if( TTx.dimensions().size() != TTy.dimensions().size() )
      throw std::invalid_argument("TensorTrainOperator: #dimensions of input/output tensor trains mismatch!");

    // check for special case: handle TTx and TTy with boundary ranks
    const bool boundaryRank = TTOp.column_dimensions().size() == TTx.dimensions().size() + 2;
    if( !boundaryRank )
    {
      // check for matching dimensions
      if( TTOp.column_dimensions() != TTx.dimensions() )
        throw std::invalid_argument("TensorTrainOperator: input tensor train dimension mismatch!");
      if( TTOp.row_dimensions() != TTy.dimensions() )
        throw std::invalid_argument("TensorTrainOperator: output tensor train dimension mismatch!");
    }
    else // boundaryRank
    {
      // special case: TTx and TTy have first/last dimension as r0 and rn

      // check dimensions
      const int nDimOp = TTOp.row_dimensions().size();
      const int nDim = TTx.dimensions().size();

      const std::vector<int> inner_col_dims{TTOp.column_dimensions().begin()+1, TTOp.column_dimensions().end()-1};
      const std::vector<int> inner_row_dims{TTOp.row_dimensions().begin()+1, TTOp.row_dimensions().end()-1};

      if( inner_col_dims != TTx.dimensions() )
        throw std::invalid_argument("TensorTrainOperator: input tensor train dimension mismatch (middle)!");
      if( inner_row_dims != TTy.dimensions() )
        throw std::invalid_argument("TensorTrainOperator: output tensor train dimension mismatch (middle)!");

      if( TTx.subTensors()[0].r1() != TTOp.column_dimensions()[0] || TTx.subTensors()[nDim-1].r2() != TTOp.column_dimensions()[nDimOp-1] )
        throw std::invalid_argument("TensorTrainOperator: input tensor train dimension mismatch (boundary rank)!");
    }

    // perform actual calculation
    for(int iDim = 0; iDim < TTx.subTensors().size(); iDim++)
    {
      const auto& subTOp = TTOp.tensorTrain().subTensors()[iDim+boundaryRank];
      const auto& subTx = TTx.subTensors()[iDim];
      auto& subTy = TTy.editableSubTensors()[iDim];

      internal::apply_contract(TTOp, iDim+boundaryRank, subTOp, subTx, subTy);
    }

    // special handling for first/last subtensor if TTx and TTy have first/last dimension as r0 and rn
    if( !boundaryRank )
      return;

    // left-most sub-tensor
    {
      const auto &subTOp0 = TTOp.tensorTrain().subTensors().front();
      auto &subTy = TTy.editableSubTensors().front();

      // contract A(0,:,*,*) * y(*,:,:)
      Tensor3<T> tmp;
      internal::apply_contract_leftBoundaryRank(TTOp, 0, subTOp0, subTy, tmp);
      std::swap(tmp, subTy);
    }

    // right-most sub-tensor
    {
      const int nDimOp = TTOp.row_dimensions().size();
      const auto &subTOpd = TTOp.tensorTrain().subTensors().back();
      auto &subTy = TTy.editableSubTensors().back();

      // contract y(:,:,*) * A(*,:,*,0)
      Tensor3<T> tmp;
      internal::apply_contract_rightBoundaryRank(subTy, TTOp, nDimOp-1, subTOpd, tmp);
      std::swap(tmp, subTy);
    }
  }

}


#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_HPP
