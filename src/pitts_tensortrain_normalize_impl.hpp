/*! @file pitts_tensortrain_normalize_impl.hpp
* @brief orthogonalization for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-17
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_NORMALIZE_IMPL_HPP
#define PITTS_TENSORTRAIN_NORMALIZE_IMPL_HPP

// includes
#include <algorithm>
#include <cassert>
#include <vector>
#include <stdexcept>
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_performance.hpp"

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
      const auto nChunks = (long long)((B.n()-1)/Chunk<T>::size+1); // remove this
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
      for(int j = 0; j < n; j++)
      {
        for(int i = 0; i < r2; i++)
        {
          T tmp[r1_];
          for(int k = 0; k < r1_; k++)
            tmp[k] = 0;
          for(int l = 0; l < r1; l++)
            for(int k = 0; k < r1_; k++)
              tmp[k] += A(k,l) * B(l,j,i);
          for(int k = 0; k < r1_; k++)
            C(k,j,i) = tmp[k];
        }
      }
    }

    //! contract Tensor3 and Tensor2 : A(:,:,*) * B(*,:)
    //!
    //! Identical to dot_contract1t but dedicated timer, so we get distinct timing results...
    //!
    template<typename T>
    void normalize_contract2(const Tensor3<T>& A, const Tensor2<T>& B, Tensor3<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto nChunks = (long long)((A.n()-1)/Chunk<T>::size+1); // remove this
      const auto r = A.r2();
      assert(A.r2() == B.r1());
      const auto r2 = B.r2();

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "nChunks", "r", "r2"},{r1, nChunks, r, r2}}, // arguments
        {{r1*nChunks*r*r2*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1*nChunks*r+r*r2)*kernel_info::Load<Chunk<T>>() + (r1*nChunks*r2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      C.resize(r1, n, r2);
      if( r1*n*r2 == 0 )
        return;

      const auto stride = A.r1()*A.n();
      using mat = Eigen::MatrixX<T>;
      Eigen::Map<const mat> mapA(&A(0,0,0), stride, r);
      const auto mapB = ConstEigenMap(B);
      Eigen::Map<mat> mapC(&C(0,0,0), stride, r2);
      mapC = mapA * mapB;
      return;

#pragma omp parallel for collapse(2) schedule(static)
      for(int j = 0; j < n; j++)
        for(int k = 0; k < r2; k++)
        {
          T tmp[r1];
          for(int i = 0; i < r1; i++)
            tmp[i] = 0;
          for(int l = 0; l < r; l++)
            for(int i = 0; i < r1; i++)
              tmp[i] += B(l,k) * A(i,j,l);
          for(int i = 0; i < r1; i++)
            C(i,j,k) = tmp[i];
        }
    }

    //! scale operation for a Tensor3
    template<typename T>
    void t3_scale(T alpha, Tensor3<T>& x)
    {
      const auto r1 = x.r1();
      const auto n = x.n();
      const auto r2 = x.r2();

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "n", "r2"},{r1, n, r2}}, // arguments
        {{r1*n*r2*kernel_info::Mult<T>()}, // flops
         {(r1*n*r2)*kernel_info::Update<T>()}} // data transfers
        );

#pragma omp parallel for collapse(3) schedule(static)
      for(int k = 0; k < r2; k++)
        for(int j = 0; j < n; j++)
          for(int i = 0; i < r1; i++)
            x(i,j,k) *= alpha;
    }

    // implement leftNormalize_range
    template<typename T>
    void leftNormalize_range(TensorTrain<T>& TT, int firstIdx, int lastIdx, T rankTolerance, int maxRank)
    {
      const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

      const auto nDim = TT.dimensions().size();

      assert(0 <= firstIdx );
      assert(lastIdx < nDim);

      // auxiliary matrix / tensor
      Tensor2<T> t2;
      std::vector<Tensor3<T>> newSubT(2);
      const std::vector<TT_Orthogonality> newSubTOrtho = {TT_Orthogonality::left, TT_Orthogonality::none};

      for(int iDim = firstIdx; iDim < lastIdx; iDim++)
      {
        const auto& subT = TT.subTensor(iDim);
        const auto& subT_next = TT.subTensor(iDim+1);

        // calculate the SVD or QR of subT(: : x :)
        unfold_left(subT, t2);

        const auto [U, Vt] = rankTolerance > 0 || maxRank < subT.r2() ?
          internal::normalize_svd(t2, true, rankTolerance / std::sqrt(T(nDim-1)), maxRank) :
          internal::normalize_qb(t2, true);

        fold_left(U, subT.n(), newSubT[0]);

        // now contract Vt(:,*) * subT_next(*,:,:)
        internal::normalize_contract1(Vt, subT_next, newSubT[1]);
        newSubT = TT.setSubTensors(iDim, std::move(newSubT), newSubTOrtho);
      }
    }

    // implement rightNormalize_range
    template<typename T>
    void rightNormalize_range(TensorTrain<T>& TT, int firstIdx, int lastIdx, T rankTolerance, int maxRank)
    {
      const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

      const auto nDim = TT.dimensions().size();

      assert(0 <= firstIdx );
      assert(lastIdx < nDim);

      // auxiliary matrix / tensor
      Tensor2<T> t2;
      std::vector<Tensor3<T>> newSubT(2);
      const std::vector<TT_Orthogonality> newSubTOrtho = {TT_Orthogonality::none, TT_Orthogonality::right};

      for(int iDim = lastIdx; iDim > firstIdx; iDim--)
      {
        const auto& subT_prev = TT.subTensor(iDim-1);
        const auto& subT = TT.subTensor(iDim);

        // calculate the SVD or QR of ( subT(: x : :) )^T
        unfold_right(subT, t2);

        const auto [U, Vt] = rankTolerance > 0 || maxRank < subT.r1() ?
          internal::normalize_svd(t2, false, rankTolerance / std::sqrt(T(nDim-1)), maxRank) :
          internal::normalize_qb(t2, false);

        fold_right(Vt, subT.n(), newSubT[1]);

        // now contract subT_prev(:,:,*) * U(*,:)
        internal::normalize_contract2(subT_prev, U, newSubT[0]);
        newSubT = TT.setSubTensors(iDim-1, std::move(newSubT), newSubTOrtho);
      }
    }

    // implement ensureLeftOrtho_range
    template<typename T>
    void ensureLeftOrtho_range(TensorTrain<T>& TT, int firstIdx, int lastIdx)
    {
      const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

      const auto nDim = TT.dimensions().size();
      assert(0 <= firstIdx );
      assert(lastIdx < nDim);

      for(int iDim = firstIdx; iDim < lastIdx; iDim++)
      {
        if( (TT.isOrthonormal(iDim) & TT_Orthogonality::left) == TT_Orthogonality::none )
        {
          leftNormalize_range(TT, iDim, lastIdx, T(0));
          return;
        }
      }
    }

    // implement ensureRightOrtho_range
    template<typename T>
    void ensureRightOrtho_range(TensorTrain<T>& TT, int firstIdx, int lastIdx)
    {
      const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

      const auto nDim = TT.dimensions().size();
      assert(0 <= firstIdx );
      assert(lastIdx < nDim);

      for(int iDim = lastIdx; iDim > firstIdx; iDim--)
      {
        if( (TT.isOrthonormal(iDim) & TT_Orthogonality::right) == TT_Orthogonality::none )
        {
          rightNormalize_range(TT, firstIdx, iDim, T(0));
          return;
        }
      }
    }
  }

  // implement TT normalize
  template<typename T>
  T normalize(TensorTrain<T>& TT, T rankTolerance, int maxRank)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    const auto nDim = TT.dimensions().size();

    auto ttOrtho = TT.isOrthogonal();
    if( ttOrtho == TT_Orthogonality::none )
    {
      internal::rightNormalize_range(TT, 0, nDim - 1, T(0));
      ttOrtho = TT_Orthogonality::right;
    }
    
    if( ttOrtho == TT_Orthogonality::right )
    {
      internal::leftNormalize_range(TT, 0, nDim-1, rankTolerance, maxRank);
      ttOrtho = TT_Orthogonality::left;
    }
    else // ttOrtho & right
    {
      internal::rightNormalize_range(TT, 0, nDim-1, rankTolerance, maxRank);
      ttOrtho = TT_Orthogonality::right;
    }

    // just calculate its norm and scale the last/first subtensor
    const int idx = (ttOrtho == TT_Orthogonality::left) ? nDim-1 : 0;
    const T nrm = internal::t3_nrm(TT.subTensor(idx));
    const T invNrm = nrm == T(0) ? T(0) : 1/nrm;
    TT.editSubTensor(idx, [invNrm](Tensor3<T>& subT){internal::t3_scale(invNrm, subT);}, ttOrtho);

    return nrm;
  }

  // implement TT leftNormalize
  template<typename T>
  T leftNormalize(TensorTrain<T>& TT, T rankTolerance, int maxRank)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    const int nDim = TT.dimensions().size();
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
    internal::leftNormalize_range(TT, 0, nDim-1, rankTolerance, maxRank);

    // just calculate its norm and scale the last subtensor
    const T nrm = internal::t3_nrm(TT.subTensor(nDim-1));
    const T invNrm = nrm == T(0) ? T(0) : 1/nrm;
    TT.editSubTensor(nDim-1, [invNrm](Tensor3<T>& subT){internal::t3_scale(invNrm, subT);}, TT_Orthogonality::left);

    return nrm;
  }


  // implement TT rightNormalize
  template<typename T>
  T rightNormalize(TensorTrain<T>& TT, T rankTolerance, int maxRank)
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    const int nDim = TT.dimensions().size();
    if( nDim <= 0)
      throw std::invalid_argument("TensorTrain #dimensions < 1!");

    internal::rightNormalize_range(TT, 0, nDim-1, rankTolerance, maxRank);

    // just calculate its norm and scale the first subtensor
    const T nrm = internal::t3_nrm(TT.subTensor(0));
    const T invNrm = nrm == T(0) ? T(0) : 1/nrm;
    TT.editSubTensor(0, [invNrm](Tensor3<T>& subT){internal::t3_scale(invNrm, subT);}, TT_Orthogonality::right);

    return nrm;
  }


}


#endif // PITTS_TENSORTRAIN_NORMALIZE_IMPL_HPP
