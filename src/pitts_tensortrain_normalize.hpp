/*! @file pitts_tensortrain_normalize.hpp
* @brief orthogonalization for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-17
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_NORMALIZE_HPP
#define PITTS_TENSORTRAIN_NORMALIZE_HPP

// includes
//#include <omp.h>
//#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cassert>
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor2 and Tensor3 : A(:,*) * B(*,:,:)
    template<typename T>
    void normalize_contract1(const Tensor2<T>& A, const Tensor3<T>& B, Tensor3<T>& C)
    {
      const auto r1 = B.r1();
      const auto n = B.n();
      const auto nChunks = B.nChunks();
      const auto r2 = B.r2();
      const auto r1_ = A.r1();
      assert(A.r2() == B.r1());
      C.resize(r1_, n, r2);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "r1_", "nChunks", "r2"},{r1, r1_, nChunks, r2}}, // arguments
        {{r1_*r1*nChunks*r2*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1*nChunks*r2+r1_*r1)*kernel_info::Load<Chunk<T>>() + (r1_*nChunks*r2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

#pragma omp parallel for collapse(2) schedule(static)
      for(int jChunk = 0; jChunk < nChunks; jChunk++)
      {
        for(int i = 0; i < r2; i++)
        {
          Chunk<T> tmp[r1_]{};
          for(int l = 0; l < r1; l++)
            for(int k = 0; k < r1_; k++)
              fmadd(A(k,l), B.chunk(l,jChunk,i), tmp[k]);
          for(int k = 0; k < r1_; k++)
            C.chunk(k,jChunk,i) = tmp[k];
        }
      }
    }

    //! contract Tensor3 and Tensor2 : A(:,:,*) * B(*,:)
    template<typename T>
    void normalize_contract2(const Tensor3<T>& A, const Tensor2<T>& B, Tensor3<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto nChunks = A.nChunks();
      const auto r = A.r2();
      assert(A.r2() == B.r1());
      const auto r2 = B.r2();

      C.resize(r1, n, r2);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "nChunks", "r", "r2"},{r1, nChunks, r, r2}}, // arguments
        {{r1*nChunks*r*r2*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1*nChunks*r+r*r2)*kernel_info::Load<Chunk<T>>() + (r1*nChunks*r2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

#pragma omp parallel for collapse(2) schedule(static)
      for(int jChunk = 0; jChunk < nChunks; jChunk++)
        for(int k = 0; k < r2; k++)
        {
          Chunk<T> tmp[r1]{};
          for(int l = 0; l < r; l++)
            for(int i = 0; i < r1; i++)
              fmadd(B(l,k), A.chunk(i,jChunk,l), tmp[i]);
          for(int i = 0; i < r1; i++)
            C.chunk(i,jChunk,k) = tmp[i];
        }
    }

    //! scale operation for a Tensor3
    template<typename T>
    void t3_scale(T alpha, Tensor3<T>& x)
    {
      const auto r1 = x.r1();
      const auto n = x.n();
      const auto nChunks = x.nChunks();
      const auto r2 = x.r2();

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "n", "r2"},{r1, n, r2}}, // arguments
        {{r1*n*r2*kernel_info::Mult<T>()}, // flops
         {(r1*n*r2)*kernel_info::Update<T>()}} // data transfers
        );

#pragma omp parallel for collapse(3) schedule(static)
      for(int k = 0; k < r2; k++)
        for(int jChunk = 0; jChunk < nChunks; jChunk++)
          for(int i = 0; i < r1; i++)
            mul(alpha, x.chunk(i,jChunk,k), x.chunk(i,jChunk,k));
    }

  }

  //! TT-rounding: truncate tensor train by two normalization sweeps (first right to left, then left to right)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TT             tensor in tensor train format, left-normalized on output
  //! @param rankTolerance  approximation tolerance
  //! @param maxRank        maximal allowed TT-rank, enforced even if this violates the rankTolerance
  //! @return               norm of the tensor
  //!
  template<typename T>
  T normalize(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max())
  {
    const auto norm = rightNormalize(TT, T(0));
    return norm * leftNormalize(TT, rankTolerance, maxRank);
  }

  //! Make all sub-tensors orthogonal sweeping left to right
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TT             tensor in tensor train format
  //! @param rankTolerance  approximation tolerance
  //! @param maxRank        maximal allowed TT-rank, enforced even if this violates the rankTolerance
  //! @return               norm of the tensor
  //!
  template<typename T>
  T leftNormalize(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max())
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    const int nDim = TT.subTensors().size();
    if( nDim <= 0)
      throw std::invalid_argument("TensorTrain #dimensions < 1!");

    // Transforms the tensor train in the following invariant way
    //
    // o--o--   --o--o
    // |  |  ...  |  |
    //
    // becomes
    //
    // q--q--   --q--*
    // |  |  ...  |  |
    //
    // where the "q" are all left-orthogonal.
    //


    // auxiliary matrix / tensor
    Tensor2<T> t2;
    Tensor3<T> t3;

    for(int iDim = 0; iDim+1 < nDim; iDim++)
    {
      auto& subT = TT.editableSubTensors()[iDim];
      const auto r2 = subT.r2();
      const auto n = subT.n();

      // calculate the SVD or QR of subT(: : x :)
      unfold_left(subT, t2);

      const auto [Q, B] = rankTolerance > 0 || maxRank < r2 ?
        internal::normalize_svd(t2, true, rankTolerance / std::sqrt(T(nDim-1)), maxRank) :
        internal::normalize_qb(t2, true);

      fold_left(Q, n, subT);

      auto& subT_next = TT.editableSubTensors()[iDim+1];
      // now contract B(:,*) * subT_next(*,:,:)
      internal::normalize_contract1(B, subT_next, t3);
      std::swap(subT_next, t3);
    }

    // just calculate its norm and scale the last subtensor
    auto& subT = TT.editableSubTensors()[nDim-1];
    const T nrm = internal::t3_nrm(subT);
    const T invNrm = nrm == T(0) ? T(0) : 1/nrm;
    internal::t3_scale(invNrm, subT);
    return nrm;
  }


  //! Make all sub-tensors orthogonal sweeping right to left
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TT             tensor in tensor train format
  //! @param rankTolerance  approximation tolerance
  //! @param maxRank        maximal allowed TT-rank, enforced even if this violates the rankTolerance
  //! @return               norm of the tensor
  //!
  template<typename T>
  T rightNormalize(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max())
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    const int nDim = TT.subTensors().size();
    if( nDim <= 0)
      throw std::invalid_argument("TensorTrain #dimensions < 1!");

    // auxiliary matrix / tensor
    Tensor2<T> t2;
    Tensor3<T> t3;

    for(int iDim = nDim-1; iDim >= 1; iDim--)
    {
      auto& subT = TT.editableSubTensors()[iDim];
      const auto r1 = subT.r1();
      const auto r2 = subT.r2();
      const auto n = subT.n();

      // calculate the SVD or QR of ( subT(: x : :) )^T
      unfold_right(subT, t2);

      const auto [U, Vt] = rankTolerance > 0 || maxRank < r2 ?
        internal::normalize_svd(t2, false, rankTolerance / std::sqrt(T(nDim-1)), maxRank) :
        internal::normalize_qb(t2, false);

      fold_right(Vt, n, subT);

      auto& subT_prev = TT.editableSubTensors()[iDim-1];
      // now contract subT_next(:,:,*) * U(*,:)
      internal::normalize_contract2(subT_prev, U, t3);
      std::swap(subT_prev, t3);
    }

    // just calculate its norm and scale the last subtensor
    auto& subT = TT.editableSubTensors()[0];
    const T nrm = internal::t3_nrm(subT);
    const T invNrm = nrm == T(0) ? T(0) : 1/nrm;
    internal::t3_scale(invNrm, subT);
    return nrm;
  }


}


#endif // PITTS_TENSORTRAIN_NORMALIZE_HPP
