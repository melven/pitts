// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_norm_impl.hpp
* @brief norms for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-09
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_NORM_IMPL_HPP
#define PITTS_TENSORTRAIN_NORM_IMPL_HPP

// includes
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor3 and Tensor2 along last dimensions: A(:,:,*) * B(:,*)
    //!
    //! Identical to dot_contract1 but could exploit symmetry of B.
    //!
    template<typename T>
    void norm2_contract1(const Tensor3<T>& A, const Tensor2<T>& B, Tensor3<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto r2 = A.r2();
      assert(A.r2() == B.r1());
      assert(B.r2() == B.r1());
      C.resize(r1, n, r2);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "n", "r2"},{r1, n, r2}}, // arguments
        {{r1*n*r2*r2*kernel_info::FMA<T>()}, // flops
         {(r1*n*r2+0.5*r2*r2)*kernel_info::Load<T>() + (r1*n*r2)*kernel_info::Store<T>()}} // data transfers
        );

#pragma omp parallel for collapse(2) schedule(static)
      for(int j = 0; j < n; j++)
      {
        for(int i = 0; i < r1; i++)
        {
          T tmp[r2];
          for(int k = 0; k < r2; k++)
            tmp[k] = 0;
          for(int l = 0; l < r2; l++)
            for(int k = 0; k < r2; k++)
              tmp[k] += B(k,l) * A(i,j,l);
          for(int k = 0; k < r2; k++)
            C(i,j,k) = tmp[k];
        }
      }
    }

    //! contract Tensor3 and Tensor3 along the last two dimensions: A(:,*,*) * B(:,*,*)
    //!
    //! Similar to dot_contract2 but actually exploits the symmetry of the result in the norm calculation.
    //!
    template<typename T>
    void norm2_contract2(const Tensor3<T>& A, const Tensor3<T>& B, Tensor2<T>& C)
    {
      const auto r1 = A.r1();
      assert(A.r1() == B.r1());
      const auto n = A.n();
      const auto r2 = A.r2();
      assert(A.n() == B.n());
      assert(A.r2() == B.r2());
      C.resize(r1,r1);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "n", "r2"},{r1, n, r2}}, // arguments
        {{0.5*r1*(r1+1)*n*r2*kernel_info::FMA<T>()}, // flops
         {(r1*n*r2+r1*n*r2)*kernel_info::Load<T>() + (r1*r1)*kernel_info::Store<T>()}} // data transfers
        );

      double tmpC[r1*r1];
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < r1; j++)
          tmpC[i*r1+j] = 0;

#pragma omp parallel reduction(+:tmpC)
{
      for(int j = 0; j < r1; j++)
        for(int i = j; i < r1; i++)
        {
          T tmp = 0;
#pragma omp for collapse(2) schedule(static) nowait
          for(int k = 0; k < n; k++)
            for(int l = 0; l < r2; l++)
              tmp += A(i,k,l) * B(j,k,l);
          tmpC[i+j*r1] = tmpC[j+i*r1] = tmp;
        }
}
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < r1; j++)
          C(i,j) = tmpC[i*r1+j];

    }

    //! 2-norm of a Tensor3 contracting Tensor3 along all dimensions sqrt(A(*,*,*) * A(*,*,*))
    template<typename T>
    T t3_nrm(const Tensor3<T>& A)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto r2 = A.r2();

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "n", "r2"},{r1, n, r2}}, // arguments
        {{r1*n*r2*kernel_info::FMA<T>()}, // flops
         {(r1*n*r2)*kernel_info::Load<T>()}} // data transfers
        );

      T result{};
#pragma omp parallel reduction(+:result)
      {
        T tmp = 0;
#pragma omp for collapse(3) schedule(static) nowait
        for(int j = 0; j < r2; j++)
          for(int k = 0; k < n; k++)
            for(int i = 0; i < r1; i++)
              tmp += A(i,k,j) * A(i,k,j);
        result = tmp;
      }

      return std::sqrt(result);
    }
  }

  //! calculate the 2-norm for a vector in tensor train format
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  T norm2(const TensorTrain<T>& TT)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    const int nDim = TT.dimensions().size();
    if( nDim <= 0)
      throw std::invalid_argument("TensorTrain #dimensions < 1!");

    // Computes the contractions
    //
    // o--o--   --o--o
    // |  |  ...  |  |
    // o--o--   --o--o
    //
    // where the top and the bottom are the same tensor.
    // Algorithm starts on the right and works like a zipper...
    //

    if( nDim == 1 )
    {
      const auto& subT = TT.subTensor(0);

      const T result = internal::t3_nrm(subT);
      return result;
    }
    
    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2;
    Tensor3<T> t3;

    // first iteration / last subtensors
    {
      const auto& subT = TT.subTensor(nDim-1);

      // only contract: subT(:,*,*) * subT(:,*,*)
      internal::norm2_contract2(subT, subT, t2);
    }

    // iterate from left to right (middle)
    for(int iDim = nDim-2; iDim >= 1; iDim--)
    {
      const auto& subT = TT.subTensor(iDim);
      
      // first contraction: subT(:,:,*) * t2(:,*)
      internal::norm2_contract1(subT, t2, t3);

      // second contraction: subT(:,*,*) * t3(:,*,*)
      internal::norm2_contract2(subT, t3, t2);
    }

    // last iteration / first subtensors
    T result;
    {
      const auto& subT = TT.subTensor(0);

      // first contraction: subT(:,:,*) * t2(:,*)
      internal::norm2_contract1(subT, t2, t3);

      // second fully contract subT(*,*,*) * t3(*,*,*)
      result = internal::t3_dot(subT, t3);
    }

    return std::sqrt(result);
  }

}


#endif // PITTS_TENSORTRAIN_NORM_IMPL_HPP
