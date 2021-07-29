/*! @file pitts_tensortrain_dot.hpp
* @brief inner products for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_DOT_HPP
#define PITTS_TENSORTRAIN_DOT_HPP

// includes
//#include <omp.h>
//#include <iostream>
#include <cassert>
#include "pitts_tensor2.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor3 and Tensor2 along last dimensions: A(:,:,*) * B(:,*)
    template<typename T>
    void dot_contract1(const Tensor3<T>& A, const Tensor2<T>& B, Tensor3<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto nChunks = A.nChunks();
      const auto r2 = A.r2();
      assert(A.r2() == B.r2());
      const auto r2_ = B.r1();
      C.resize(r1, n, r2_);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "nChunks", "r2", "r2_"},{r1, nChunks, r2, r2_}}, // arguments
        {{r1*nChunks*r2*r2_*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1*nChunks*r2+r2*r2_)*kernel_info::Load<Chunk<T>>() + (r1*nChunks*r2_)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

#pragma omp parallel for collapse(2) schedule(static)
      for(int jChunk = 0; jChunk < nChunks; jChunk++)
      {
        for(int i = 0; i < r1; i++)
        {
          Chunk<T> tmp[r2_]{};
          for(int l = 0; l < r2; l++)
            for(int k = 0; k < r2_; k++)
              fmadd(B(k,l), A.chunk(i,jChunk,l), tmp[k]);
          for(int k = 0; k < r2_; k++)
            C.chunk(i,jChunk,k) = tmp[k];
        }
      }
    }

    //! contract Tensor3 and Tensor3 along the last two dimensions: A(:,*,*) * B(:,*,*)
    template<typename T>
    void dot_contract2(const Tensor3<T>& A, const Tensor3<T>& B, Tensor2<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto nChunks = A.nChunks();
      const auto r2 = A.r2();
      assert(A.r2() == B.r2());
      const auto r1_ = B.r1();
      C.resize(r1,r1_);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "r1_", "nChunks", "r2"},{r1, r1_, nChunks, r2}}, // arguments
        {{r1*r1_*nChunks*r2*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1*nChunks*r2+r1_*nChunks*r2)*kernel_info::Load<Chunk<T>>() + (r1_*r1)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      double tmpC[r1*r1_];
      for(int j = 0; j < r1_; j++)
        for(int i = 0; i < r1; i++)
          tmpC[i+j*r1] = 0;

#pragma omp parallel reduction(+:tmpC)
{
      for(int j = 0; j < r1_; j++)
        for(int i = 0; i < r1; i++)
        {
          Chunk<T> tmp{};
#pragma omp for collapse(2) schedule(static) nowait
          for(int kChunk = 0; kChunk < nChunks; kChunk++)
            for(int l = 0; l < r2; l++)
              fmadd(A.chunk(i,kChunk,l), B.chunk(j,kChunk,l), tmp);
          tmpC[i+j*r1] = sum(tmp);
        }
}
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < r1_; j++)
          C(i,j) = tmpC[i+j*r1];

    }
  }

  //! calculate the inner product for two vectors in tensor train format
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  T dot(const TensorTrain<T>& TT1, const TensorTrain<T>& TT2)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    if( TT1.dimensions() != TT2.dimensions() )
      throw std::invalid_argument("TensorTrain dot dimensions mismatch!");

    // Computes the contractions
    //
    // o--o--   --o--o
    // |  |  ...  |  |
    // o--o--   --o--o
    //
    // Algorithm starts on the right and works like a zipper...
    //
    
    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2(1,1);
    t2(0,0) = T(1);
    Tensor3<T> t3;

    // iterate from left to right
    const int nDim = TT1.subTensors().size();
    for(int iDim = nDim-1; iDim >= 0; iDim--)
    {
      const auto& subT1 = TT1.subTensors()[iDim];
      const auto& subT2 = TT2.subTensors()[iDim];

      // first contraction: subT1(:,:,*) * t2(:,*)
      internal::dot_contract1(subT1, t2, t3);

      // second contraction: subT2(:,*,*) * t3(:,*,*)
      internal::dot_contract2(subT2, t3, t2);
    }
    return t2(0,0);
  }

}


#endif // PITTS_TENSORTRAIN_DOT_HPP
