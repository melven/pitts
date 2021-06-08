/*! @file pitts_tensortrain_norm.hpp
* @brief norms for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-09
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_NORM_HPP
#define PITTS_TENSORTRAIN_NORM_HPP

// includes
//#include <omp.h>
//#include <iostream>
#include <cmath>
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
    void norm2_contract1(const Tensor3<T>& A, const Tensor2<T>& B, Tensor3<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto nChunks = A.nChunks();
      const auto r2 = A.r2(); // == B.r1() == B.r2()
      C.resize(r1, n, r2);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "nChunks", "r2"},{r1, nChunks, r2}}, // arguments
        {{r1*nChunks*r2*r2*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1*nChunks*r2+r2*r2)*kernel_info::Load<Chunk<T>>() + (r1*nChunks*r2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

#pragma omp parallel for collapse(2) schedule(static)
      for(int jChunk = 0; jChunk < nChunks; jChunk++)
      {
        for(int i = 0; i < r1; i++)
        {
          Chunk<T> tmp[r2]{};
          for(int l = 0; l < r2; l++)
            for(int k = 0; k < r2; k++)
              fmadd(B(k,l), A.chunk(i,jChunk,l), tmp[k]);
          for(int k = 0; k < r2; k++)
            C.chunk(i,jChunk,k) = tmp[k];
        }
      }
    }

    //! contract Tensor3 and Tensor3 along the last two dimensions: A(:,*,*) * B(:,*,*)
    //!
    //! exploits the symmetry of the result in the norm calculation
    //!
    template<typename T>
    void norm2_contract2(const Tensor3<T>& A, const Tensor3<T>& B, Tensor2<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto nChunks = A.nChunks();
      const auto r2 = A.r2(); // == B.r1() == B.r2()
      C.resize(r1,r1);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "nChunks", "r2"},{r1, nChunks, r2}}, // arguments
        {{0.5*r1*(r1+1)*nChunks*r2*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1*nChunks*r2+r1*nChunks*r2)*kernel_info::Load<Chunk<T>>() + (r1*r1)*kernel_info::Store<Chunk<T>>()}} // data transfers
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
          Chunk<T> tmp{};
#pragma omp for schedule(static) nowait
          for(int kChunk = 0; kChunk < nChunks; kChunk++)
            for(int l = 0; l < r2; l++)
              fmadd(A.chunk(i,kChunk,l), B.chunk(j,kChunk,l), tmp);
          tmpC[i+j*r1] = tmpC[j+i*r1] = sum(tmp);
        }
}
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < r1; j++)
          C(i,j) = tmpC[i*r1+j];

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

    // Computes the contractions
    //
    // o--o--   --o--o
    // |  |  ...  |  |
    // o--o--   --o--o
    //
    // where the top and the bottom are the same tensor.
    // Algorithm starts on the right and works like a zipper...
    //
    
    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2(1,1);
    t2(0,0) = T(1);
    Tensor3<T> t3;

    // iterate from right to left
    const int nDim = TT.subTensors().size();
    for(int iDim = nDim-1; iDim >= 0; iDim--)
    {
      const auto& subT = TT.subTensors()[iDim];
      const auto r1 = subT.r1();
      const auto r2 = subT.r2();
      const auto n = subT.n();

      // first contraction: subT(:,:,*) * t2(:,*)
      internal::norm2_contract1(subT, t2, t3);

      // second contraction: subT(:,*,*) * t3(:,*,*)
      internal::norm2_contract2(subT, t3, t2);
    }

    return std::sqrt(t2(0,0));
  }

}


#endif // PITTS_TENSORTRAIN_NORM_HPP
