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
#include <memory>
#include <cassert>
#include <vector>
#include "pitts_tensortrain_operator.hpp"
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

      y.resize(rA1*r1, n, rA2*r2);
      const int nChunks = y.nChunks();

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

      // copy Aop to obtain a better memory layout... (chunk-wise access)
      std::unique_ptr<Chunk<T>[]> tmpAop{new Chunk<T>[nChunks*rA1*rA2*m]};
      for(int i2 = 0; i2 < rA2; i2++)
        for(int kChunk = 0; kChunk < nChunks; kChunk++)
          for(int i1 = 0; i1 < rA1; i1++)
            for(int l = 0; l < m; l++)
            {
              const int k_begin = kChunk*Chunk<T>::size;
              const int k_end = std::min<int>(n, (kChunk+1)*Chunk<T>::size);
              Chunk<T> tmp;
              for(int k = 0; k < Chunk<T>::size; k++)
                tmp[k] = k+k_begin < k_end ? Aop(i1,k+k_begin+n*l,i2) : T(0);
              tmpAop[l+i1*m+kChunk*(m*rA1)+i2*(m*rA1*nChunks)] = tmp;
            }


      // resulting y is the biggest array and this is easily memory-bound, so order loops according to memory layout of y
#pragma omp parallel for collapse(3) schedule(static)
      for(int j = 0; j < rA2*r2; j++)
        for(int kChunk = 0; kChunk < nChunks; kChunk++)
          for(int i = 0; i < rA1*r1; i++)
          {
            const int i1 = i / r1;
            const int j1 = i % r1;
            const int i2 = j / r2;
            const int j2 = j % r2;

            Chunk<T> tmp = Chunk<T>{};
            for(int l = 0; l < m; l++)
              fmadd(x(j1,l,j2), tmpAop[l+i1*m+kChunk*(m*rA1)+i2*(m*rA1*nChunks)], tmp);
            y.chunk(i,kChunk,j) = tmp;
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

      const int nChunks = y.nChunks();
#pragma omp parallel for collapse(2) schedule(static)
      for(int j = 0; j < r2; j++)
        for(int kChunk = 0; kChunk < nChunks; kChunk++)
        {
          for(int iy = 0; iy < n; iy++)
          {
            Chunk<T> tmp{};
            for(int ix = 0; ix < r1; ix++)
            {
              const int l = ix % m;
              const int j2 = ix / m;
              fmadd(Aop(0,iy+n*l,j2), x.chunk(ix,kChunk,j), tmp);
            }
            y.chunk(iy,kChunk,j) = tmp;
          }
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

      const int nChunks = y.nChunks();

#pragma omp parallel for collapse(2) schedule(static)
      for(int jy = 0; jy < n; jy++)
        for(int kChunk = 0; kChunk < nChunks; kChunk++)
        {
          for(int i = 0; i < r1; i++)
          {
            Chunk<T> tmp{};
            for(int jx = 0; jx < r2; jx++)
            {
              const int l = jx % m;
              const int i2 = jx / m;
              fmadd(Aop(i2,jy+l*n,0), x.chunk(i,kChunk,jx), tmp);
            }
            y.chunk(i,kChunk,jy) = tmp;
          }
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


#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_HPP
