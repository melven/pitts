/*! @file pitts_tensortrain_dot_impl.hpp
* @brief inner products for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_DOT_IMPL_HPP
#define PITTS_TENSORTRAIN_DOT_IMPL_HPP

// includes
#include <cassert>
#include <stdexcept>
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
          Chunk<T> tmp[r2_];
          for(int k = 0; k < r2_; k++)
            tmp[k] = Chunk<T>{};
          for(int l = 0; l < r2; l++)
            for(int k = 0; k < r2_; k++)
              fmadd(B(k,l), A.chunk(i,jChunk,l), tmp[k]);
          for(int k = 0; k < r2_; k++)
            C.chunk(i,jChunk,k) = tmp[k];
        }
      }
    }

    //! contract Tensor3 and Tensor2 along last and first dimensions: A(:,:,*) * B(*,:)
    template<typename T>
    void dot_contract1t(const Tensor3<T>& A, const Tensor2<T>& B, Tensor3<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto nChunks = A.nChunks();
      const auto r2 = A.r2();
      assert(A.r2() == B.r1());
      const auto r2_ = B.r2();
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
          Chunk<T> tmp[r2_];
          for(int k = 0; k < r2_; k++)
            tmp[k] = Chunk<T>{};
          for(int l = 0; l < r2; l++)
            for(int k = 0; k < r2_; k++)
              fmadd(B(l,k), A.chunk(i,jChunk,l), tmp[k]);
          for(int k = 0; k < r2_; k++)
            C.chunk(i,jChunk,k) = tmp[k];
        }
      }
    }

    //! contract Tensor3 and Tensor2 along first dimensions: A(*,:) * B(*,:,:)
    template<typename T>
    void reverse_dot_contract1(const Tensor2<T>& A, const Tensor3<T>& B, Tensor3<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = B.n();
      const auto nChunks = B.nChunks();
      const auto r2 = B.r2();
      assert(A.r1() == B.r1());
      const auto r1_ = A.r2();
      C.resize(r1_, n, r2);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "nChunks", "r2", "r1_"},{r1, nChunks, r2, r1_}}, // arguments
        {{r1*nChunks*r2*r1_*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1*nChunks*r2+r1*r1_)*kernel_info::Load<Chunk<T>>() + (r1_*nChunks*r2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

#pragma omp parallel for collapse(2) schedule(static)
      for(int jChunk = 0; jChunk < nChunks; jChunk++)
      {
        for (int k = 0; k < r2; k++)
        {
          Chunk<T> tmp[r1_];
          for (int i = 0; i < r1_; i++)
            tmp[i] = Chunk<T>{};
          for (int l = 0; l < r1; l++)
            for (int i = 0; i < r1_; i++)
              fmadd(A(l,i), B.chunk(l,jChunk,k), tmp[i]);
          for (int i = 0; i < r1_; i++)
            C.chunk(i, jChunk, k) = tmp[i];
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

      // spatial blocking
      constexpr int bs = 10;

      if( r1*r1_ > bs*bs*nChunks )
      {
#pragma omp parallel for collapse(2)
        for(int jb = 0; jb < r1_; jb+=bs)
          for(int ib = 0; ib < r1; ib+=bs)
          {
            Chunk<T> tmp[bs*bs];
            for(int i = 0; i < bs*bs; i++)
              tmp[i] = Chunk<T>{};
            for(int l = 0; l < r2; l++)
              for(int kChunk = 0; kChunk < nChunks; kChunk++)
                for(int j = jb; j < std::min((int)r1_, jb+bs); j++)
                  for(int i = ib; i < std::min((int)r1, ib+bs); i++)
                    fmadd(A.chunk(i,kChunk,l), B.chunk(j,kChunk,l), tmp[i-ib+(j-jb)*bs]);
            for(int j = jb; j < std::min((int)r1_, jb+bs); j++)
              for(int i = ib; i < std::min((int)r1, ib+bs); i++)
                C(i,j) = sum(tmp[i-ib+(j-jb)*bs]);
          }
      }
      else // r1*r1_ < bs*bs*nChunks
      {
          for(int jb = 0; jb < r1_; jb+=bs)
            for(int ib = 0; ib < r1; ib+=bs)
            {
              T tmp[bs*bs];
              for(int j = 0; j < bs; j++)
                for(int i = 0; i < bs; i++)
                  tmp[i+j*bs] = T(0);
#pragma omp parallel reduction(+:tmp)
              {
                Chunk<T> tmpC[bs*bs];
                for(int i = 0; i < bs*bs; i++)
                  tmpC[i] = Chunk<T>{};
#pragma omp for collapse(2) schedule(static)
              for(int l = 0; l < r2; l++)
                for(int kChunk = 0; kChunk < nChunks; kChunk++)
                {
                  for(int j = jb; j < std::min((int)r1_, jb+bs); j++)
                    for(int i = ib; i < std::min((int)r1, ib+bs); i++)
                      fmadd(A.chunk(i,kChunk,l), B.chunk(j,kChunk,l), tmpC[i-ib+(j-jb)*bs]);
                }
              for(int j = jb; j < std::min((int)r1_, jb+bs); j++)
                for(int i = ib; i < std::min((int)r1, ib+bs); i++)
                  tmp[i-ib+(j-jb)*bs] = sum(tmpC[i-ib+(j-jb)*bs]);
              }
              for(int j = jb; j < std::min((int)r1_, jb+bs); j++)
                for(int i = ib; i < std::min((int)r1, ib+bs); i++)
                  C(i,j) = tmp[i-ib+(j-jb)*bs];
            }
        /*
        T tmpC[r1*r1_];
        for(int j = 0; j < r1_; j++)
          for(int i = 0; i < r1; i++)
            tmpC[i+j*r1] = T(0);
#pragma omp parallel reduction(+:tmpC)
        {
          for(int jb = 0; jb < r1_; jb+=bs)
            for(int ib = 0; ib < r1; ib+=bs)
            {
              Chunk<T> tmp[bs*bs];
              for(int i = 0; i < bs*bs; i++)
                tmp[i] = Chunk<T>{};
#pragma omp for collapse(2) schedule(static) nowait
              for(int l = 0; l < r2; l++)
                for(int kChunk = 0; kChunk < nChunks; kChunk++)
                  for(int j = jb; j < std::min(r1_, jb+bs); j++)
                    for(int i = ib; i < std::min(r1, ib+bs); i++)
                      fmadd(A.chunk(i,kChunk,l), B.chunk(j,kChunk,l), tmp[i-ib+(j-jb)*bs]);
              for(int j = jb; j < std::min(r1_, jb+bs); j++)
                for(int i = ib; i < std::min(r1, ib+bs); i++)
                  tmpC[i+j*r1] += sum(tmp[i-ib+(j-jb)*bs]);
            }
          for (int j = 0; j < r1_; j++)
            for (int i = 0; i < r1; i++)
              C(i, j) = tmpC[i + j * r1];
        }
        */
      }

    }

    //! contract Tensor3 and Tensor3 along the first two dimensions: A(*,*,:) * B(*,*,:)
    template<typename T>
    void reverse_dot_contract2(const Tensor3<T>& A, const Tensor3<T>& B, Tensor2<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto nChunks = A.nChunks();
      const auto rA2 = A.r2();
      assert(A.r1() == B.r1());
      assert(A.n() == B.n());
      const auto rB2 = B.r2();
      C.resize(rA2,rB2);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "nChunks", "rA2", "rB2"},{r1, nChunks, rA2, rB2}}, // arguments
        {{r1*nChunks*rA2*rB2*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1*nChunks*rA2+r1*nChunks*rB2)*kernel_info::Load<Chunk<T>>() + (rA2*rB2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      //double tmpC[rA2*rB2];
      //for(int j = 0; j < rA2; j++)
      //  for(int i = 0; i < rB2; i++)
      //    tmpC[i+j*rA2] = 0;

//#pragma omp parallel reduction(+:tmpC)
{
      for (int j = 0; j < rB2; j++)
        for (int i = 0; i < rA2; i++)
        {
          Chunk<T> tmp{};
//#pragma omp for collapse(2) schedule(static) nowait
          for(int kChunk = 0; kChunk < nChunks; kChunk++)
            for(int l = 0; l < r1; l++)
              fmadd(A.chunk(l,kChunk,i), B.chunk(l,kChunk,j), tmp);
          //tmpC[i+j*rA2] = sum(tmp);
          C(i,j) = sum(tmp);
        }
}
      //for(int i = 0; i < rA2; i++)
      //  for(int j = 0; j < rB2; j++)
      //    C(i,j) = tmpC[i+j*rA2];
    }


    //! contract Tensor3 and Tensor3 along all dimensions: A(*,*,*) * B(*,*,*)
    template<typename T>
    T t3_dot(const Tensor3<T>& A, const Tensor3<T>& B)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto nChunks = A.nChunks();
      const auto r2 = A.r2();
      assert(A.r1() == B.r1());
      assert(A.n() == B.n());
      assert(A.r2() == B.r2());

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "n", "r2"},{r1, n, r2}}, // arguments
        {{r1*n*r2*kernel_info::FMA<T>()}, // flops
         {(2*r1*n*r2)*kernel_info::Load<T>()}} // data transfers
        );

      T result{};
#pragma omp parallel reduction(+:result)
      {
        Chunk<T> tmp{};
#pragma omp for collapse(3) schedule(static) nowait
        for(int j = 0; j < r2; j++)
          for(int kChunk = 0; kChunk < nChunks; kChunk++)
            for(int i = 0; i < r1; i++)
              fmadd(A.chunk(i,kChunk,j), B.chunk(i,kChunk,j), tmp);
        result = sum(tmp);
      }

      return result;
    }
  }

  // implement TT dot
  template<typename T>
  T dot(const TensorTrain<T>& TT1, const TensorTrain<T>& TT2)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    if( TT1.dimensions() != TT2.dimensions() )
      throw std::invalid_argument("TensorTrain dot dimensions mismatch!");

    const int nDim = TT1.dimensions().size();
    if( nDim <= 0)
      throw std::invalid_argument("TensorTrain #dimensions < 1!");

    // we can handle first r1 != 1 and last r2 != 1 but the ranks have to match...
    if( TT1.subTensor(     0).r1() != TT2.subTensor(     0).r1() ||
        TT1.subTensor(nDim-1).r2() != TT2.subTensor(nDim-1).r2()  )
      throw std::invalid_argument("TensorTrain boundary ranks mismatch!");

    // Computes the contractions
    //
    // o--o--   --o--o
    // |  |  ...  |  |
    // o--o--   --o--o
    //
    // Algorithm starts on the right and works like a zipper...
    //

    if( nDim == 1 )
    {
      const auto& subT1 = TT1.subTensor(0);
      const auto& subT2 = TT2.subTensor(0);

      const T result = internal::t3_dot(subT1, subT2);
      return result;
    }
    
    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2;
    Tensor3<T> t3;

    // first iteration / last subtensors
    {
      const auto& subT1 = TT1.subTensor(nDim-1);
      const auto& subT2 = TT2.subTensor(nDim-1);

      // only contract: subT2(:,*,*) * subT1(:,*,*)
      internal::dot_contract2(subT2, subT1, t2);
    }

    // iterate from left to right (middle)
    for(int iDim = nDim-2; iDim >= 1; iDim--)
    {
      const auto& subT1 = TT1.subTensor(iDim);
      const auto& subT2 = TT2.subTensor(iDim);
      
      if( t2.r1() > t2.r2() )
      {
        // first contraction: subT2(:,:,*) * t2(*,:)
        internal::dot_contract1t(subT2, t2, t3);

        // second contraction: t3(:,*,*) * subT1(:,*,*)
        internal::dot_contract2(t3, subT1, t2);
      }
      else // t2.r1() < t2.r2()
      {
        // first contraction: subT1(:,:,*) * t2(:,*)
        internal::dot_contract1(subT1, t2, t3);

        // second contraction: subT2(:,*,*) * t3(:,*,*)
        internal::dot_contract2(subT2, t3, t2);
      }
    }

    // last iteration / first subtensors
    T result;
    {
      const auto& subT1 = TT1.subTensor(0);
      const auto& subT2 = TT2.subTensor(0);

      // first contraction: subT1(:,:,*) * t2(:,*)
      internal::dot_contract1(subT1, t2, t3);

      // second fully contract subT2(*,*,*) * t3(*,*,*)
      result = internal::t3_dot(subT2, t3);
    }

    return result;
  }

}


#endif // PITTS_TENSORTRAIN_DOT_IMPL_HPP
