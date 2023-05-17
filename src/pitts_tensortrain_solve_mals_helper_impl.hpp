/*! @file pitts_tensortrain_solve_mals_helper_impl.hpp
* @brief helper functionality for PITTS::solveMALS
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-05-05
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SOLVE_MALS_HELPER_IMPL_HPP
#define PITTS_TENSORTRAIN_SOLVE_MALS_HELPER_IMPL_HPP

// includes
#include <cassert>
#include <vector>
#include "pitts_tensortrain_solve_mals_helper.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_eigen.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_axpby.hpp"
#include "pitts_multivector_scale.hpp"
#include "pitts_multivector_dot.hpp"
#include "pitts_multivector_norm.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_tensortrain_from_dense.hpp"
#include "pitts_gmres.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! dedicated helper functions for solveMALS
    namespace solve_mals
    {
      // calculate next part of Ax from right to left or discard last part
      template<typename T>
      void update_right_Ax(const TensorTrainOperator<T> TTOpA, const TensorTrain<T>& TTx, int firstIdx, int lastIdx,
                           std::vector<Tensor3<T>>& right_Ax, std::vector<Tensor3<T>>& right_Ax_ortho, std::vector<Tensor2<T>>& right_Ax_ortho_M)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int nDim = TTx.dimensions().size();
        assert(TTx.dimensions() == TTOpA.column_dimensions());
        assert(0 <= firstIdx);
        assert(firstIdx <= lastIdx+1);
        assert(lastIdx == nDim-1);
        assert(right_Ax_ortho.size() >= right_Ax.size());
        assert(right_Ax_ortho_M.size() == right_Ax_ortho.size());

        // discard old entries in right_Ax_ortho*
        while(right_Ax_ortho.size() > right_Ax.size())
          right_Ax_ortho.pop_back();
        while(right_Ax_ortho_M.size() > right_Ax.size())
          right_Ax_ortho_M.pop_back();

        // calculate new entries in right_Ax when sweeping right-to-left
        for(int iDim = lastIdx - right_Ax.size(); iDim >= firstIdx; iDim--)
        {
          const auto &subTx = TTx.subTensor(iDim);
          const auto &subTOpA = TTOpA.tensorTrain().subTensor(iDim);
          const auto n = subTOpA.n() / subTx.n();
          {
            Tensor3<T> subTAx;
            internal::apply_contract(TTOpA, iDim, subTOpA, subTx, subTAx);
            right_Ax.emplace_back(std::move(subTAx));
          }

          Tensor2<T> t2Ax;
          Tensor3<T> subTAx;
          if( right_Ax_ortho_M.empty() )
          {
            unfold_right(right_Ax.back(), t2Ax);
          }
          else
          {
            internal::normalize_contract2(right_Ax.back(), right_Ax_ortho_M.back(), subTAx);
            unfold_right(subTAx, t2Ax);
          }
          auto [B,Qt] = internal::normalize_qb(t2Ax, false);
          right_Ax_ortho_M.emplace_back(std::move(B));
          fold_right(Qt, n, subTAx);
          right_Ax_ortho.emplace_back(std::move(subTAx));
        }

        // discard old entries in right_Ax when sweeping left-to-right
        for(int iDim = lastIdx - right_Ax.size(); iDim+1 < firstIdx; iDim++)
        {
          right_Ax.pop_back();
          right_Ax_ortho.pop_back();
          right_Ax_ortho_M.pop_back();
        }
      }

      // calculate next part of Ax from left to right or discard last part
      template<typename T>
      void update_left_Ax(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx, int firstIdx, int lastIdx,
                          std::vector<Tensor3<T>>& left_Ax, std::vector<Tensor3<T>>& left_Ax_ortho, std::vector<Tensor2<T>>& left_Ax_ortho_M)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int nDim = TTx.dimensions().size();
        assert(TTx.dimensions() == TTOpA.column_dimensions());
        assert(0 == firstIdx);
        assert(firstIdx-1 <= lastIdx);
        assert(lastIdx < nDim);

        assert(left_Ax_ortho.size() >= left_Ax.size());
        assert(left_Ax_ortho_M.size() == left_Ax_ortho.size());

        // discard old entries in left_Ax_ortho*
        while(left_Ax_ortho.size() > left_Ax.size())
          left_Ax_ortho.pop_back();
        while(left_Ax_ortho_M.size() > left_Ax.size())
          left_Ax_ortho_M.pop_back();

        // calculate new entries in left_Ax when sweeping left-to-right
        for(int iDim = firstIdx + left_Ax.size(); iDim <= lastIdx; iDim++)
        {
          const auto& subTOpA = TTOpA.tensorTrain().subTensor(iDim);
          const auto& subTx = TTx.subTensor(iDim);
          const auto n = subTOpA.n() / subTx.n();
          {
            Tensor3<T> subTAx;
            internal::apply_contract(TTOpA, iDim, subTOpA, subTx, subTAx);
            left_Ax.emplace_back(std::move(subTAx));
          }

          Tensor2<T> t2Ax;
          Tensor3<T> subTAx;
          if( left_Ax_ortho_M.empty() )
          {
            unfold_left(left_Ax.back(), t2Ax);
          }
          else
          {
            internal::normalize_contract1(left_Ax_ortho_M.back(), left_Ax.back(), subTAx);
            unfold_left(subTAx, t2Ax);
          }
          auto [Q, B] = internal::normalize_qb(t2Ax, true);
          left_Ax_ortho_M.emplace_back(std::move(B));
          fold_left(Q, n, subTAx);
          left_Ax_ortho.emplace_back(std::move(subTAx));
        }

        // discard old entries in left_Ax when sweeping right-to-left
        for(int iDim = firstIdx + left_Ax.size(); iDim-1 > lastIdx; iDim--)
        {
          left_Ax.pop_back();
          left_Ax_ortho.pop_back();
          left_Ax_ortho_M.pop_back();
        }
      }

      // set up TT operator for the projection (assumes given TTx is correctly orthogonalized)
      template<typename T>
      TensorTrainOperator<T> setupProjectionOperator(const TensorTrain<T>& TTx, SweepIndex swpIdx)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const auto& rowDims = TTx.dimensions();
        const int nDim = rowDims.size();
        std::vector<int> colDims(nDim);
        for(int iDim = 0; iDim < nDim; iDim++)
        {
          if( iDim+1 == swpIdx.leftDim() )
            colDims[iDim] = TTx.subTensor(iDim).r2();
          else if( iDim-1 == swpIdx.rightDim() )
            colDims[iDim] = TTx.subTensor(iDim).r1();
          else if( iDim >= swpIdx.leftDim() && iDim <= swpIdx.rightDim() )
            colDims[iDim] = rowDims[iDim];
          else // < leftDim or > rightDim
            colDims[iDim] = 1;
        }

        TensorTrainOperator<T> TTOp(rowDims, colDims);
        std::vector<Tensor3<T>> subTensors(nDim);
        std::vector<TT_Orthogonality> ortho(nDim);
        for(int iDim = 0; iDim < nDim; iDim++)
        {
          Tensor3<T>& subTOp = subTensors[iDim];
          const Tensor3<T>& subTx = TTx.subTensor(iDim);
          if( iDim+1 == swpIdx.leftDim() )
          {
            subTOp.resize(subTx.r1(),rowDims[iDim]*colDims[iDim],1);
            for(int i = 0; i < subTx.r1(); i++)
              for(int j = 0; j < rowDims[iDim]; j++)
                for(int k = 0; k < colDims[iDim]; k++)
                  subTOp(i ,TTOp.index(iDim, j, k), 0) = subTx(i, j, k);
            ortho[iDim] = TT_Orthogonality::none;
          }
          else if( iDim-1 == swpIdx.rightDim() )
          {
            subTOp.resize(1,rowDims[iDim]*colDims[iDim],subTx.r2());
            for(int i = 0; i < rowDims[iDim]; i++)
              for(int j = 0; j < colDims[iDim]; j++)
                for(int k = 0; k < subTx.r2(); k++)
                  subTOp(0, TTOp.index(iDim, i, j), k) = subTx(j, i, k);
            ortho[iDim] = TT_Orthogonality::none;
          }
          else if( iDim >= swpIdx.leftDim() && iDim <= swpIdx.rightDim() )
          {
            // create identity operator
            subTOp.resize(1,rowDims[iDim]*colDims[iDim],1);
            subTOp.setConstant(T(0));
            for(int i = 0; i < rowDims[iDim]; i++)
              subTOp(0, TTOp.index(iDim, i, i), 0) = T(1);
            ortho[iDim] = TT_Orthogonality::none;
          }
          else // < leftDim or > rightDim
          {
            copy(subTx, subTOp);
            ortho[iDim] = TTx.isOrthonormal(iDim);
          }
        }
        TTOp.tensorTrain().setSubTensors(0, std::move(subTensors), ortho);

        return TTOp;
      }


      template<typename T>
      TensorTrain<T> calculatePetrovGalerkinProjection(TensorTrainOperator<T>& TTAv, SweepIndex swpIdx, const TensorTrain<T>& TTx, bool symmetrize)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        internal::ensureLeftOrtho_range(TTAv.tensorTrain(), 0, swpIdx.leftDim());
        internal::ensureRightOrtho_range(TTAv.tensorTrain(), swpIdx.rightDim(), swpIdx.nDim()-1);

        std::vector<Tensor3<T>> subTensors;
        std::vector<TT_Orthogonality> ortho;
        subTensors.reserve(swpIdx.nDim()+2);
        ortho.reserve(swpIdx.nDim()+2);
        for(int iDim = 0; iDim < swpIdx.leftDim()-1; iDim++)
        {
          assert(TTAv.column_dimensions().at(iDim) == 1);
          Tensor3<T> subT;
          copy(TTAv.tensorTrain().subTensor(iDim), subT);
          subTensors.emplace_back(std::move(subT));
          ortho.push_back(TTAv.tensorTrain().isOrthonormal(iDim));
        }
        if( swpIdx.leftDim()-1 >= 0 )
        {
          const int iDim = swpIdx.leftDim()-1;
          const int n = TTAv.row_dimensions()[iDim];
          const int rleft = TTAv.column_dimensions()[iDim];
          auto [t3a, t3b] = split(TTAv.tensorTrain().subTensor(iDim), n, rleft, false, true);
          subTensors.emplace_back(std::move(t3a));
          ortho.push_back(TT_Orthogonality::left);
          subTensors.emplace_back(std::move(t3b));
          ortho.push_back(TT_Orthogonality::none);
        }
        for(int iDim = swpIdx.leftDim(); iDim <= swpIdx.rightDim(); iDim++)
        {
          Tensor3<T> subT;
          copy(TTAv.tensorTrain().subTensor(iDim), subT);
          subTensors.emplace_back(std::move(subT));
          ortho.push_back(TTAv.tensorTrain().isOrthonormal(iDim));
        }
        if( swpIdx.rightDim()+1 < swpIdx.nDim() )
        {
          const int iDim = swpIdx.rightDim()+1;
          const int n = TTAv.row_dimensions()[iDim];
          const int rright = TTAv.column_dimensions()[iDim];
          auto [t3a, t3b] = split(TTAv.tensorTrain().subTensor(iDim), rright, n, true, false);
          subTensors.emplace_back(std::move(t3a));
          ortho.push_back(TT_Orthogonality::none);
          subTensors.emplace_back(std::move(t3b));
          ortho.push_back(TT_Orthogonality::right);
        }
        for(int iDim = swpIdx.rightDim()+2; iDim < swpIdx.nDim(); iDim++)
        {
          assert(TTAv.column_dimensions().at(iDim) == 1);
          Tensor3<T> subT;
          copy(TTAv.tensorTrain().subTensor(iDim), subT);
          subTensors.emplace_back(std::move(subT));
          ortho.push_back(TTAv.tensorTrain().isOrthonormal(iDim));
        }

        const int nDim = subTensors.size();
        TensorTrain<T> TTw(std::move(subTensors), ortho);
        // make left- and right-orthogonal
        if( swpIdx.leftDim()-1 >= 0 )
        {
          internal::ensureLeftOrtho_range(TTw, 0, swpIdx.leftDim());
        }
        if( swpIdx.rightDim()+1 < swpIdx.nDim() )
        {
          const int rightDim = nDim - (swpIdx.nDim()-swpIdx.rightDim());
          internal::ensureRightOrtho_range(TTw, rightDim, nDim-1);
        }

        // we now need to truncate such that the ranks at leftDim and rightDim match with TTx
        if( swpIdx.leftDim()-1 >= 0 )
        {
          const int rleft = TTw.dimensions()[swpIdx.leftDim()];
          internal::ensureLeftOrtho_range(TTw, 0, swpIdx.leftDim()-1);
          internal::ensureRightOrtho_range(TTw, swpIdx.leftDim()-1, nDim-1);
          internal::leftNormalize_range(TTw, swpIdx.leftDim()-1, swpIdx.leftDim(), T(0), rleft);

          // try to make it as symmetric as possible (locally)
          if( symmetrize )
          {
            if( swpIdx.leftDim()+1 < nDim )
              internal::ensureLeftOrtho_range(TTw, 0, swpIdx.leftDim()+1);
            Tensor2<T> Q(rleft, rleft);
            const Tensor3<T>& subT = TTw.subTensor(swpIdx.leftDim());
            if( subT.r1() != rleft )
              throw std::runtime_error("solveMALS: (case not implemented) couldn't make local problem quadratic...");
            // SVD subT(:,:,0) to make more symmetric
            for(int i = 0; i < rleft; i++)
              for(int j = 0; j < rleft; j++)
                Q(i,j) = subT(i,j,0);
#if EIGEN_VERSION_AT_LEAST(3,4,90)
            const Eigen::BDCSVD<Eigen::MatrixX<T>, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(ConstEigenMap(Q));
#else
            const Eigen::BDCSVD<Eigen::MatrixX<T>> svd(ConstEigenMap(Q), Eigen::ComputeFullU | Eigen::ComputeFullV);
#endif
            // std::cout << "  leftSymmetrize: " << svd.rank() << " nonzero sing. val. of " << svd.rows() << "\n";
            // if( svd.rank() > 1 )
            //   std::cout << "  distinct singular values: " << (svd.singularValues().head(svd.rank()-1) - svd.singularValues().tail(svd.rank()-1)).minCoeff() << "\n";
            EigenMap(Q) = svd.matrixV() * svd.matrixU().transpose();
            std::vector<Tensor3<T>> newSubT(2);
            internal::normalize_contract1(Q, subT, newSubT[1]);
            internal::dot_contract1(TTw.subTensor(swpIdx.leftDim()-1), Q, newSubT[0]);
            std::vector<TT_Orthogonality> newOrtho(2);
            newOrtho[0] = TTw.isOrthonormal(swpIdx.leftDim()-1);
            newOrtho[1] = TTw.isOrthonormal(swpIdx.leftDim());
            TTw.setSubTensors(swpIdx.leftDim()-1, std::move(newSubT), newOrtho);
            // TODO: check that symmetry stays the same (and not later destroyed by another operation!!)
          }
        }
        if( swpIdx.rightDim()+1 < swpIdx.nDim() )
        {
          const int rightDim = nDim - (swpIdx.nDim()-swpIdx.rightDim());
          const int rright = TTw.dimensions()[rightDim];
          internal::ensureLeftOrtho_range(TTw, 0, rightDim+1);
          internal::ensureRightOrtho_range(TTw, rightDim+1, nDim-1);
          internal::rightNormalize_range(TTw, rightDim, rightDim+1, T(0), rright);

          // try to make it as symmetric as possible (locally)
          {
            if( rightDim-1 >= 0 )
              internal::ensureRightOrtho_range(TTw, rightDim-1, nDim-1);
            Tensor2<T> Q(rright, rright);
            const Tensor3<T>& subT = TTw.subTensor(rightDim);
            if( subT.r2() != rright )
              throw std::runtime_error("solveMALS: (case not implemented) couldn't make local problem quadratic...");
            // SVD of subT(0,:,:) to make it more symmetric
            for(int i = 0; i < rright; i++)
              for(int j = 0; j < rright; j++)
                Q(i,j) = subT(0,i,j);
#if EIGEN_VERSION_AT_LEAST(3,4,90)
          const Eigen::BDCSVD<Eigen::MatrixX<T>, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(ConstEigenMap(Q));
#else
          const Eigen::BDCSVD<Eigen::MatrixX<T>> svd(ConstEigenMap(Q), Eigen::ComputeFullU | Eigen::ComputeFullV);
#endif
            // std::cout << "  rightSymmetrize: " << svd.rank() << " nonzero sing. val. of " << svd.rows() << "\n";
            // if( svd.rank() > 1 )
            //   std::cout << "  distinct singular values: " << (svd.singularValues().head(svd.rank()-1) - svd.singularValues().tail(svd.rank()-1)).minCoeff() << "\n";
            EigenMap(Q) = svd.matrixV() * svd.matrixU().transpose();
            std::vector<Tensor3<T>> newSubT(2);
            internal::normalize_contract2(subT, Q, newSubT[0]);
            internal::reverse_dot_contract1(Q, TTw.subTensor(rightDim+1), newSubT[1]);
            std::vector<TT_Orthogonality> newOrtho(2);
            newOrtho[0] = TTw.isOrthonormal(rightDim);
            newOrtho[1] = TTw.isOrthonormal(rightDim+1);
            TTw.setSubTensors(rightDim, std::move(newSubT), newOrtho);
            // TODO: check that symmetry stays the same (and not later destroyed by another operation!!)
          }
        }

        // return TTw;
        // store in same format as TTx would be (for testing, we could return TTw here!)
        subTensors = std::vector<Tensor3<T>>(swpIdx.nDim());
        ortho.resize(swpIdx.nDim());
        for(int iDim = 0; iDim < swpIdx.leftDim(); iDim++)
        {
          copy(TTw.subTensor(iDim), subTensors[iDim]);
          ortho[iDim] = TTw.isOrthonormal(iDim);
        }
        for(int iDim = swpIdx.rightDim()+1; iDim < swpIdx.nDim(); iDim++)
        {
          copy(TTw.subTensor(nDim - (swpIdx.nDim()-iDim)), subTensors[iDim]);
          ortho[iDim] = TTw.isOrthonormal(nDim - (swpIdx.nDim()-iDim));
        }
        // middle has just the correct dimensions...
        for(int iDim = swpIdx.leftDim(); iDim <= swpIdx.rightDim(); iDim++)
        {
          const int r1 = iDim == 0 ? 1 : subTensors[iDim-1].r2();
          const int r2 = iDim < swpIdx.rightDim() || iDim+1 == swpIdx.nDim() ? 1 : subTensors[iDim+1].r1();
          const int n = TTx.dimensions()[iDim];
          subTensors[iDim].resize(r1, n, r2);
          ortho[iDim] = TT_Orthogonality::none;
        }
        TensorTrain<T> TTw_(std::move(subTensors), ortho);

        return TTw_;
      }

      // calculate next part of v^Tw from right to left or discard last part
      //
      // Like TT dot product fused with TT apply but allows to store all intermediate results.
      //
      template<typename T>
      void update_right_vTw(const RightPartialTT<T>& TTv, const RightPartialTT<T>& TTw, int firstIdx, int lastIdx, std::vector<Tensor2<T>>& right_vTw)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        assert(0 <= firstIdx);
        assert(firstIdx <= lastIdx+1);
        //assert(lastIdx == nDim-1);

        // first call? right_vTAw should at least contain a 1x1 one tensor
        if( right_vTw.empty() )
        {
          right_vTw.emplace_back(Tensor2<T>{1,1});
          right_vTw.back()(0,0) = T(1);
        }

        // calculate new entries in right_vTw when sweeping right-to-left
        for(int iDim = lastIdx - (right_vTw.size()-1); iDim >= firstIdx; iDim--)
        {
          const auto& subTv = TTv.subTensorFromRight(lastIdx-iDim);
          const auto& subTw = TTw.subTensorFromRight(lastIdx-iDim);

          // first contraction: subTw(:,:,*) * prev_t2(:,*)
          Tensor3<T> t3_tmp;
          internal::dot_contract1(subTw, right_vTw.back(), t3_tmp);

          // second contraction: subTv(:,*,*) * t3_tmp(:,*,*)
          Tensor2<T> t2;
          internal::dot_contract2(subTv, t3_tmp, t2);
          right_vTw.emplace_back(std::move(t2));
        }

        // discard old entries in right_vTw when sweeping left-to-right
        for(int iDim = lastIdx - (right_vTw.size()-1); iDim+1 < firstIdx; iDim++)
          right_vTw.pop_back();
      }

      // calculate next part of v^Tw from left to right or discard last part
      //
      // Like TT dot product fused with TT apply but allows to store all intermediate results.
      //
      template<typename T>
      void update_left_vTw(const LeftPartialTT<T>& TTv, const LeftPartialTT<T>& TTw, int firstIdx, int lastIdx, std::vector<Tensor2<T>>& left_vTw)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        assert(0 == firstIdx);
        assert(firstIdx-1 <= lastIdx);
        //assert(lastIdx < nDim);

        // first call? left_vTw should at least contain a 1x1 one tensor
        if( left_vTw.empty() )
        {
          left_vTw.emplace_back(Tensor2<T>{1,1});
          left_vTw.back()(0,0) = T(1);
        }

        // calculate new entries in left_vTw when sweeping left-to-right
        for(int iDim = firstIdx + (left_vTw.size()-1); iDim <= lastIdx; iDim++)
        {
          const auto& subTv = TTv.subTensorFromLeft(iDim);
          const auto& subTw = TTw.subTensorFromLeft(iDim);

          // first contraction: prev_t2(*,:) * subTw(*,:,:)
          Tensor3<T> t3_tmp;
          internal::reverse_dot_contract1(left_vTw.back(), subTw, t3_tmp);

          // second contraction: t3(*,*,:) * subTv(*,*,:)
          Tensor2<T> t2;
          internal::reverse_dot_contract2(t3_tmp, subTv, t2);
          left_vTw.emplace_back(std::move(t2));
        }

        // discard old entries in left_vTw when sweeping right-to-left
        for(int iDim = firstIdx + (left_vTw.size()-1); iDim-1 > lastIdx; iDim--)
          left_vTw.pop_back();
      }

      // calculate the local RHS tensor-train for (M)ALS
      template<typename T>
      TensorTrain<T> calculate_local_rhs(int iDim, int nMALS, optional_cref<Tensor2<T>> left_vTb, const TensorTrain<T>& TTb, optional_cref<Tensor2<T>> right_vTb)
      {
        std::vector<Tensor3<T>> subT_b(nMALS);
        for(int i = 0; i < nMALS; i++)
          copy(TTb.subTensor(iDim+i), subT_b[i]);

        Tensor3<T> t3_tmp;
        // first contract: tt_b_right(:,:,*) * right_vTb(:,*)
        if( right_vTb )
        {
          std::swap(subT_b.back(), t3_tmp);
          internal::dot_contract1<T>(t3_tmp, *right_vTb, subT_b.back());
        }

        // then contract: left_vTb(*,:) * tt_b_left(*,:,:)
        if( left_vTb )
        {
          std::swap(subT_b.front(), t3_tmp);
          internal::reverse_dot_contract1<T>(*left_vTb, t3_tmp, subT_b.front());
        }

        TensorTrain<T> tt_b(std::move(subT_b));

        return tt_b;
      }

      // calculate the local initial solutiuon in TT format for (M)ALS
      template<typename T>
      TensorTrain<T> calculate_local_x(int iDim, int nMALS, const TensorTrain<T>& TTx)
      {
        std::vector<Tensor3<T>> subT_x(nMALS);
        for(int i = 0; i < nMALS; i++)
          copy(TTx.subTensor(iDim+i), subT_x[i]);

        TensorTrain<T> tt_x(std::move(subT_x));

        return tt_x;
      }

      template<typename T>
      void copy_op_left(const Tensor2<T>& t2, const int rAl, Tensor3<T>& t3)
      {
          const int r1 = t2.r2();
          const int r1_ = t2.r1() / rAl;

          const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
              {{"r1_", "rAl", "r1"}, {r1_, rAl, r1}},   // arguments
              {{r1_*r1*rAl*kernel_info::NoOp<T>()},    // flops
              {r1_*r1*rAl*kernel_info::Store<T>() + r1_*r1*rAl*kernel_info::Load<T>()}}  // data
              );

          t3.resize(1,r1_*r1,rAl);

#pragma omp parallel for collapse(3) schedule(static) if(r1*r1_*rAl > 500)
          for(int i = 0; i < r1; i++)
            for(int j = 0; j < r1_; j++)
              for(int k = 0; k < rAl; k++)
                t3(0,i+j*r1,k) = t2(j+k*r1_,i);
      }

      template<typename T>
      void copy_op_right(const Tensor2<T>& t2, const int rAr, Tensor3<T>& t3)
      {
          const int r2 = t2.r1();
          const int r2_ = t2.r2() / rAr;

          const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
              {{"r2", "rAr", "r2_"}, {r2, rAr, r2_}},   // arguments
              {{r2*r2_*rAr*kernel_info::NoOp<T>()},    // flops
              {r2*r2_*rAr*kernel_info::Store<T>() + r2*r2_*rAr*kernel_info::Load<T>()}}  // data
              );

          t3.resize(rAr,r2_*r2,1);

#pragma omp parallel for collapse(3) schedule(static) if(r2*r2_*rAr > 500)
          for(int i = 0; i < r2; i++)
            for(int j = 0; j < r2_; j++)
              for(int k = 0; k < rAr; k++)
                t3(k,i+j*r2,0) = t2(i,j+k*r2_);
      }

      // calculate the local linear operator in TT format for (M)ALS
      template<typename T>
      TensorTrainOperator<T> calculate_local_op(int iDim, int nMALS, optional_cref<Tensor2<T>> left_vTAx, const TensorTrainOperator<T>& TTOp, optional_cref<Tensor2<T>> right_vTAx)
      {
        int nDim = nMALS + (bool)left_vTAx + (bool)right_vTAx;
        std::vector<Tensor3<T>> subT_localOp(nDim);
        for(int i = 0; i < nMALS; i++)
          copy(TTOp.tensorTrain().subTensor(iDim+i), subT_localOp[(bool)left_vTAx+i]);
        if( left_vTAx )
          copy_op_left<T>(*left_vTAx, subT_localOp[1].r1(), subT_localOp.front());
        if( right_vTAx )
          copy_op_right<T>(*right_vTAx, subT_localOp[nMALS].r2(), subT_localOp.back());

        std::vector<int> localRowDims(nDim), localColDims(nDim);
        if( left_vTAx )
        {
          localRowDims.front() = left_vTAx->get().r2();
          localColDims.front() = subT_localOp[0].n() / localRowDims.front();
        }
        for(int i = 0; i < nMALS; i++)
        {
          localRowDims[i+(bool)left_vTAx] = TTOp.row_dimensions()[iDim+i];
          localColDims[i+(bool)left_vTAx] = TTOp.column_dimensions()[iDim+i];
        }
        if( right_vTAx )
        {
          localRowDims.back() = right_vTAx->get().r1();
          localColDims.back()  = subT_localOp.back().n() / localRowDims.back();
        }

        TensorTrainOperator<T> localTTOp(localRowDims, localColDims);
        localTTOp.tensorTrain().setSubTensors(0, std::move(subT_localOp));

        return localTTOp;
      }

      template<typename T>
      T solveDenseGMRES(const TensorTrainOperator<T>& tt_OpA, bool symmetric, const TensorTrain<T>& tt_b, TensorTrain<T>& tt_x,
                        int maxRank, int maxIter, T absTol, T relTol, const std::string& outputPrefix, bool verbose)
      {
        using arr = Eigen::ArrayX<T>;
        const int nDim = tt_x.dimensions().size();
        // GMRES with dense vectors...
        MultiVector<T> mv_x, mv_rhs;
        toDense(tt_x, mv_x);
        toDense(tt_b, mv_rhs);

        // absolute tolerance is not invariant wrt. #dimensions
        const arr localRes = GMRES<arr>(tt_OpA, symmetric, mv_rhs, mv_x, maxIter, arr::Constant(1, absTol), arr::Constant(1, relTol), outputPrefix, verbose);

        const auto r_left = tt_x.subTensor(0).r1();
        const auto r_right = tt_x.subTensor(nDim-1).r2();
        TensorTrain<T> new_tt_x = fromDense(mv_x, mv_rhs, tt_x.dimensions(), absTol/nDim, maxRank, false, r_left, r_right);
        std::swap(tt_x, new_tt_x);

        return localRes(0);
      }
    }
  }

}


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_HELPER_IMPL_HPP
