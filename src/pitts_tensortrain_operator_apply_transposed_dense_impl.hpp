// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_operator_apply_transposed_dense_impl.hpp
* @brief apply a tensor train operator to a dense tensor
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-12
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_APPLY_TRANSPOSED_DENSE_IMPL_HPP
#define PITTS_TENSORTRAIN_OPERATOR_APPLY_TRANSPOSED_DENSE_IMPL_HPP

// includes
#include <cmath>
#include <numeric>
#include <cassert>
#include <stdexcept>
#include "pitts_tensortrain_operator.hpp"
#include "pitts_multivector.hpp"
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
    //! contract Tensor3-Operator (e.g. rank-4 tensor) and (flattened) dense tensor
    template<typename T>
    void applyT_dense_contract(const TensorTrainOperator<T>& TTOp, int iDim, const Tensor3<T>& Aop, const MultiVector<T>& x, MultiVector<T>& y)
    {
      constexpr auto partial_products = [](const auto& v, int middle)
      {
        assert(middle < v.size());
        const long n_left = std::accumulate(v.begin(), v.begin()+middle, 1, std::multiplies<long>());
        const long n_right = std::accumulate(v.begin()+middle+1, v.end(), 1, std::multiplies<long>());
        return std::make_pair(n_left, n_right);
      };
      
      const long nCols = x.cols();
      const long n = TTOp.column_dimensions()[iDim];
      const long m = TTOp.row_dimensions()[iDim];
      const long r1 = Aop.r1();
      const long r2 = Aop.r2();
      const int nDim = TTOp.row_dimensions().size();
      const auto [n_left, n_right] = partial_products(TTOp.column_dimensions(), iDim);
      const auto [m_left, m_right] = partial_products(TTOp.row_dimensions(), iDim);
      
      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"nCols", "m_right", "r2", "n", "n_left", "m", "r1"},{nCols, m_right, r2, n, n_left, m, r1}}, // arguments
        {{nCols*m_right*r2*n*n_left*m*r1*kernel_info::FMA<T>()}, // flops
         {(r1*n*m*r2+nCols*n_left*r1*m*m_right)*kernel_info::Load<Chunk<T>>() + (nCols*n_left*n*r2*m_right)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      // x should have n_left * r1 * m * m_right #rows
      // y should get n_left * n * r2 * m_right #rows
      // both stored column major in this order...
      assert(n_left * r1 * m * m_right == x.rows());

      y.resize(n_left * n * r2 * m_right, nCols);

      for(int iCol = 0; iCol < nCols; iCol++)
      {
        for(int jr = 0; jr < m_right; jr++)
          for(int k2 = 0; k2 < r2; k2++)
            for(int i = 0; i < n; i++)
              for(int jl = 0; jl < n_left; jl++)
              {
                T tmp{};
                for(int j = 0; j < m; j++)
                  for(int k1 = 0; k1 < r1; k1++)
                    tmp += Aop(k1,TTOp.index(iDim,j,i),k2) * x(jl + k1*n_left + j*n_left*r1 + jr*n_left*r1*m, iCol);
                y(jl + i*n_left + k2*n_left*n + jr*n_left*n*r2, iCol) = tmp;
              }
      }
    }
  }

  // implement TT Op apply transposed to dense
  template<typename T>
  void applyT(const TensorTrainOperator<T>& TTOp, const MultiVector<T>& MVx, MultiVector<T>& MVy)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrainOperator<T>>();

    // check for matching dimensions
    const long nTotal = std::accumulate(TTOp.row_dimensions().begin(), TTOp.row_dimensions().end(), 1, std::multiplies<long>());
    if( nTotal != MVx.rows() )
      throw std::invalid_argument("TensorTrainOperator: input tensor dimension mismatch!");

    MultiVector<T> mv_tmp;

    // perform actual calculation
    const int nDim = TTOp.row_dimensions().size();
    for(int iDim = 0; iDim < nDim; iDim++)
    {
      const auto& subTOp = TTOp.tensorTrain().subTensor(iDim);
      const auto& x = (iDim == 0) ? MVx : MVy;
      auto& y = mv_tmp;

      internal::applyT_dense_contract(TTOp, iDim, subTOp, x, y);

      std::swap(y, MVy);
    }
  }

}


#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_TRANSPOSED_DENSE_IMPL_HPP
