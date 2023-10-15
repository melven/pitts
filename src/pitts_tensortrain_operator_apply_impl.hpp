// Copyright (c) 2021 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_operator_apply_impl.hpp
* @brief apply a tensor train operator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2021-02-12
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_APPLY_IMPL_HPP
#define PITTS_TENSORTRAIN_OPERATOR_APPLY_IMPL_HPP

// includes
#include <cmath>
#include <memory>
#include <cassert>
#include <vector>
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_performance.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_eigen.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor3-Operator (e.g. rank-4 tensor) and Tensor3 along middle dimension: A(:,:,*,:) * x(:,*,:)
    template<typename T>
    void apply_contract([[maybe_unused]] const TensorTrainOperator<T>& TTOp, [[maybe_unused]] int iDim, const Tensor3<T>& Aop, const Tensor3<T>& x, Tensor3<T>& y)
    {
      const auto rA1 = Aop.r1();
      const auto rA2 = Aop.r2();
      const auto r1 = x.r1();
      const auto r2 = x.r2();
      const auto m = x.n();
      const auto n = Aop.n() / m;

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"rA1", "n", "m", "rA2", "r1", "r2"},{rA1, n, m, rA2, r1, r2}}, // arguments
        {{rA1*rA2*n*m*r1*r2*kernel_info::FMA<T>()}, // flops
         {(rA1*m*n*rA2+r1*m*r2)*kernel_info::Load<Chunk<T>>() + (rA1*r1*n*rA2*r2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      y.resize(rA1*r1, n, rA2*r2);

      using mat = Eigen::MatrixX<T>;
#pragma omp parallel for schedule(static) collapse(2) if(r2*rA2 > 50)
      for(long long i2 = 0; i2 < rA2; i2++)
        for(long long j2 = 0; j2 < r2; j2++)
        {
          Eigen::Map<const mat> mapX(&x(0,0,j2), r1, m);
          Eigen::Map<const mat> mapA(&Aop(0,0,i2), rA1*n, m);
          Eigen::Map<mat> mapY(&y(0,0,j2+i2*r2), r1, rA1*n);
          mapY.noalias() = mapX * mapA.transpose();
        }
    }

    //! contract Tensor3-Operator (e.g. rank-4 tensor) and Tensor3 along some dimensions: A(0,:,*,*) * x(*,:,:)
    template<typename T>
    void apply_contract_leftBoundaryRank([[maybe_unused]] const TensorTrainOperator<T>& TTOp, [[maybe_unused]] int iDim, const Tensor3<T>& Aop, const Tensor3<T>& x, Tensor3<T>& y)
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

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"rA2", "r1", "xn", "r2", "n", "m"},{rA2, r1, xn, r2, n, m}}, // arguments
        {{m*r1*xn*r2*kernel_info::FMA<T>()}, // flops
         {(m*n*rA2+r1*xn*r2)*kernel_info::Load<Chunk<T>>() + (m*xn*r2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      y.resize(n, xn, r2);

      if( r2 > 20 )
      {
#pragma omp parallel for schedule(static)
        for(long long i = 0; i < r2; i++)
        {
          using mat = Eigen::MatrixX<T>;
          Eigen::Map<const mat> mapA(&Aop(0,0,0), n, r1);
          Eigen::Map<const mat> mapX(&x(0,0,i), r1, xn);
          Eigen::Map<mat> mapY(&y(0,0,i), n, xn);
          mapY.noalias() = mapA * mapX;
        }
      }
      else
      {
        // small version
        using mat = Eigen::MatrixX<T>;
        Eigen::Map<const mat> mapA(&Aop(0,0,0), n, r1);
        Eigen::Map<const mat> mapX(&x(0,0,0), r1, xn*r2);
        Eigen::Map<mat> mapY(&y(0,0,0), n, xn*r2);
        mapY.noalias() = mapA * mapX;
      }
    }

    //! contract Tensor3-Operator (e.g. rank-4 tensor) and Tensor3 along some dimensions: x(:,:,*) * A(*,:,*,0)
    template<typename T>
    void apply_contract_rightBoundaryRank(const Tensor3<T>& x, [[maybe_unused]] const TensorTrainOperator<T>& TTOp, [[maybe_unused]] int iDim, const Tensor3<T>& Aop, Tensor3<T>& y)
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

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"rA1", "r1", "xn", "r2", "n", "m"},{rA1, r1, xn, r2, n, m}}, // arguments
        {{r1*xn*r2*m*kernel_info::FMA<T>()}, // flops
         {(rA1*n*m+r1*xn*r2)*kernel_info::Load<Chunk<T>>() + (r1*xn*m)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      y.resize(r1, xn, n);

      // copy Aop to buffer, so we can call standard GEMM
      using mat = Eigen::MatrixX<T>;
      mat tmpA(m*rA1, n);
#pragma omp parallel for collapse(2) schedule(static) if(n*rA1 > 20)
      for(long long k = 0; k < n; k++)
        for(long long j = 0; j < rA1; j++)
          for(long long i = 0; i < m; i++)
            tmpA(i+j*m,k) = Aop(j,k+i*n,0);
      
      Eigen::Map<const mat> viewX(&x(0,0,0), r1*xn, m*rA1);
      Eigen::Map<mat> viewY(&y(0,0,0), r1*xn, n);
      viewY.noalias() = viewX * tmpA;
    }
  }

  // implemen TT Op apply
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

      if( TTx.subTensor(0).r1() != TTOp.column_dimensions()[0] || TTx.subTensor(nDim-1).r2() != TTOp.column_dimensions()[nDimOp-1] )
        throw std::invalid_argument("TensorTrainOperator: input tensor train dimension mismatch (boundary rank)!");
    }


    // perform actual calculation
    {
      std::vector<Tensor3<T>> newSubTy(TTy.dimensions().size());
      for(int iDim = 0; iDim < TTx.dimensions().size(); iDim++)
      {
        const auto& subTOp = TTOp.tensorTrain().subTensor(iDim+boundaryRank);
        const auto& subTx = TTx.subTensor(iDim);
        const auto& subTy = TTy.subTensor(iDim);

        internal::apply_contract(TTOp, iDim+boundaryRank, subTOp, subTx, newSubTy[iDim]);
      }
      TTy.setSubTensors(0, std::move(newSubTy));
    }

    // special handling for first/last subtensor if TTx and TTy have first/last dimension as r0 and rn
    if( !boundaryRank )
      return;

    // left-most sub-tensor
    Tensor3<T> tmp;
    {
      const auto &subTOp0 = TTOp.tensorTrain().subTensor(0);
      const auto &subTy = TTy.subTensor(0);

      // contract A(0,:,*,*) * y(*,:,:)
      internal::apply_contract_leftBoundaryRank(TTOp, 0, subTOp0, subTy, tmp);
      tmp = TTy.setSubTensor(0, std::move(tmp));
    }

    // right-most sub-tensor
    {
      const int nDimOp = TTOp.row_dimensions().size();
      const int nDimy = TTy.dimensions().size();
      const auto &subTOpd = TTOp.tensorTrain().subTensor(nDimOp-1);
      const auto &subTy = TTy.subTensor(nDimy-1);

      // contract y(:,:,*) * A(*,:,*,0)
      internal::apply_contract_rightBoundaryRank(subTy, TTOp, nDimOp-1, subTOpd, tmp);
      tmp = TTy.setSubTensor(nDimy-1, std::move(tmp));
    }
  }

}


#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_IMPL_HPP
