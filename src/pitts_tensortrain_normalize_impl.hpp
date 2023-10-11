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
      const auto r2 = B.r2();
      const auto r1_ = A.r1();
      assert(A.r2() == B.r1());

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "r1_", "n", "r2"},{r1, r1_, n, r2}}, // arguments
        {{r1_*r1*n*r2*kernel_info::FMA<T>()}, // flops
         {(r1_*r1+r1*n*r2)*kernel_info::Load<T>() + (r1_*n*r2)*kernel_info::Store<T>()}} // data transfers
        );

      C.resize(r1_, n, r2);
      EigenMap(unfold_right(C)).noalias() = ConstEigenMap(A) * ConstEigenMap(unfold_right(B));
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
      const auto r = A.r2();
      assert(A.r2() == B.r1());
      const auto r2 = B.r2();

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "n", "r", "r2"},{r1, n, r, r2}}, // arguments
        {{r1*n*r*r2*kernel_info::FMA<T>()}, // flops
         {(r1*n*r+r*r2)*kernel_info::Load<T>() + (r1*n*r2)*kernel_info::Store<T>()}} // data transfers
        );

      C.resize(r1, n, r2);

      EigenMap(unfold_left(C)).noalias() = ConstEigenMap(unfold_left(A)) * ConstEigenMap(B);
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
      std::vector<Tensor3<T>> newSubT(2);
      const std::vector<TT_Orthogonality> newSubTOrtho = {TT_Orthogonality::left, TT_Orthogonality::none};

      T rankTol = rankTolerance / std::sqrt(T(nDim-1));

      for(int iDim = firstIdx; iDim < lastIdx; iDim++)
      {
        const auto& subT = TT.subTensor(iDim);
        const auto& subT_next = TT.subTensor(iDim+1);

        T oldFrobeniusNorm;

        // calculate the SVD or QR of subT(: : x :)
        // for the SVD, use a relative tolerance in the Frobenius norm in the first call,
        // then use an absolute tolerance multiplied with the norm obtained from the first call
        auto [U, Vt] = rankTol > 0 || maxRank < subT.r2() ?
          internal::normalize_svd(unfold_left(subT), true, rankTol, maxRank, iDim!=firstIdx, true, &oldFrobeniusNorm) :
          internal::normalize_qb(unfold_left(subT), true);
        
        if( iDim == firstIdx && rankTol > 0 )
          rankTol *= oldFrobeniusNorm;

        newSubT[0] = fold_left(std::move(U), subT.n());

// for controlling the error:
//bool wasRightOrtho = (TT.isOrthonormal(iDim+1) & TT_Orthogonality::right) != TT_Orthogonality::none;
//Eigen::MatrixX<T> VtV;
//if( wasRightOrtho && (rankTolerance > 0 || maxRank < subT.r2()) )
//{
//  VtV = ConstEigenMap(Vt) * ConstEigenMap(Vt).transpose();
//}

        // now contract Vt(:,*) * subT_next(*,:,:)
        internal::normalize_contract1(Vt, subT_next, newSubT[1]);
        newSubT = TT.setSubTensors(iDim, std::move(newSubT), newSubTOrtho);

//if( wasRightOrtho && (rankTolerance > 0 || maxRank < subT.r2()) )
//{
//  unfold_right(TT.subTensor(iDim+1), t2);
//  std::cout << "est. error: " << (ConstEigenMap(t2) * ConstEigenMap(t2).transpose() - VtV).array().abs().maxCoeff()/VtV(0,0) << "\n";
//}
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
      std::vector<Tensor3<T>> newSubT(2);
      const std::vector<TT_Orthogonality> newSubTOrtho = {TT_Orthogonality::none, TT_Orthogonality::right};

      T rankTol = rankTolerance / std::sqrt(T(nDim-1));

      for(int iDim = lastIdx; iDim > firstIdx; iDim--)
      {
        const auto& subT_prev = TT.subTensor(iDim-1);
        const auto& subT = TT.subTensor(iDim);

        T oldFrobeniusNorm;

        // calculate the SVD or QR of ( subT(: x : :) )^T
        // for the SVD, use a relative tolerance in the Frobenius norm in the first call,
        // then use an absolute tolerance multiplied with the norm obtained from the first call
        auto [U, Vt] = rankTol > 0 || maxRank < subT.r1() ?
          internal::normalize_svd(unfold_right(subT), false, rankTol, maxRank, iDim!=lastIdx, true, &oldFrobeniusNorm) :
          internal::normalize_qb(unfold_right(subT), false);
        
        if( iDim == lastIdx && rankTol > 0 )
          rankTol *= oldFrobeniusNorm;

        newSubT[1] = fold_right(std::move(Vt), subT.n());

// for controlling the error:
//bool wasLeftOrtho = (TT.isOrthonormal(iDim-1) & TT_Orthogonality::left) != TT_Orthogonality::none;
//Eigen::MatrixX<T> UtU;
//if( wasLeftOrtho && (rankTolerance > 0 || maxRank < subT.r1()) )
//{
//  UtU = ConstEigenMap(U).transpose() * ConstEigenMap(U);
//}

        // now contract subT_prev(:,:,*) * U(*,:)
        internal::normalize_contract2(subT_prev, U, newSubT[0]);
        newSubT = TT.setSubTensors(iDim-1, std::move(newSubT), newSubTOrtho);

//if( wasLeftOrtho && (rankTolerance > 0 || maxRank < subT.r1()) )
//{
//  unfold_left(TT.subTensor(iDim-1), t2);
//  std::cout << "est. error: " << (ConstEigenMap(t2).transpose() * ConstEigenMap(t2) - UtU).array().abs().maxCoeff()/UtU(0,0) << "\n";
//}
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
    // special case for boundary rank: this is also not correct as boundary rank should be handled like an additional dimension...
    //const auto lastOrtho = TT.subTensor(nDim-1).r2() != 1 ? TT_Orthogonality::none : TT_Orthogonality::left;
    //TT.editSubTensor(nDim-1, [invNrm](Tensor3<T>& subT){internal::t3_scale(invNrm, subT);}, lastOrtho);
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
    // special case for boundary rank: this is also not correct as boundary rank should be handled like an additional dimension...
    //const auto firstOrtho = TT.subTensor(0).r1() != 1 ? TT_Orthogonality::none : TT_Orthogonality::right;
    //TT.editSubTensor(0, [invNrm](Tensor3<T>& subT){internal::t3_scale(invNrm, subT);}, firstOrtho);
    TT.editSubTensor(0, [invNrm](Tensor3<T>& subT){internal::t3_scale(invNrm, subT);}, TT_Orthogonality::right);

    return nrm;
  }


}


#endif // PITTS_TENSORTRAIN_NORMALIZE_IMPL_HPP
