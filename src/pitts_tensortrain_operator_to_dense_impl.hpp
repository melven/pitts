// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_operator_to_dense_impl.hpp
* @brief conversion from the tensor-train operator format into a dense matrix
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-03-13
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_TO_DENSE_IMPL_HPP
#define PITTS_TENSORTRAIN_OPERATOR_TO_DENSE_IMPL_HPP

// includes
#include "pitts_tensortrain_operator_to_dense.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // implement TTOp toDense
  template<typename T>
  Tensor2<T> toDense(const TensorTrainOperator<T>& TTOp)
  {
    // timer
    const auto timer = PITTS::timing::createScopedTimer<TensorTrainOperator<T>>();

    const int nDim = TTOp.row_dimensions().size();

    if( nDim == 0 )
      return Tensor2<T>(0,0);

    Tensor3<T> tmp(1,1,1);
    tmp.setConstant(T(1));

    for(int iDim = 0; iDim < nDim; iDim++)
    {
      const Tensor3<T>& subT = TTOp.tensorTrain().subTensor(iDim);

      // contract middle dimension with first dimension tmp(1:nl,*,1:ml) * subT(*,1:n*m,1:rnext)
      const int nl = tmp.r1();
      const int ml = tmp.r2();
      const int n = TTOp.row_dimensions()[iDim];
      const int m = TTOp.column_dimensions()[iDim];
      const int r = subT.r1();
      assert(r == tmp.n());
      const int rnext = subT.r2();
      Tensor3<T> tmp_next(nl*n,subT.r2(),ml*m);
      for(int i = 0; i < nl*n; i++)
        for(int j = 0; j < rnext; j++)
          for(int k = 0; k < ml*m; k++)
          {
            tmp_next(i,j,k) = 0;
            for(int l = 0; l < r; l++)
              tmp_next(i,j,k) += tmp(i % nl, l, k % ml) * subT(l, i/nl + (k/ml)*n, j);
          }
      std::swap(tmp_next, tmp);
    }
    
    return unfold_left(std::move(tmp));
  }
}


#endif // PITTS_TENSORTRAIN_OPERATOR_TO_DENSE_IMPL_HPP
