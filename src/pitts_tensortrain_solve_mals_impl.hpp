/*! @file pitts_tensortrain_solve_mals_impl.hpp
* @brief MALS algorithm for solving (non-)symmetric linear systems in TT format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-28
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SOLVE_MALS_IMPL_HPP
#define PITTS_TENSORTRAIN_SOLVE_MALS_IMPL_HPP

// includes
//#include <omp.h>
//#include <iostream>
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>
#include "pitts_tensortrain_solve_mals.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor3_combine.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_operator_apply_dense.hpp"
#include "pitts_tensortrain_operator_apply_transposed.hpp"
#include "pitts_tensortrain_operator_apply_op.hpp"
#include "pitts_tensortrain_operator_apply_transposed_op.hpp"
#include "pitts_tensortrain_operator_to_dense.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_tensortrain_from_dense.hpp"
#include "pitts_tensortrain_solve_gmres.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_axpby.hpp"
#include "pitts_multivector_norm.hpp"
#include "pitts_multivector_dot.hpp"
#include "pitts_multivector_scale.hpp"
#include "pitts_multivector_eigen_adaptor.hpp"
#include "pitts_gmres.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"
#include "pitts_tensortrain_sweep_index.hpp"
#include "pitts_performance.hpp"
#ifndef NDEBUG
#include "pitts_tensortrain_debug.hpp"
#include "pitts_tensortrain_operator_debug.hpp"
#endif

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    
    //! dedicated helper functions for solveMALS
    namespace solve_mals
    {
      //! calculate next part of Ax from right to left or discard last part
      template<typename T>
      void update_right_Ax(const TensorTrainOperator<T> TTOpA, const TensorTrain<T>& TTx, int firstIdx, int lastIdx, std::vector<Tensor3<T>>& right_Ax)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int nDim = TTx.dimensions().size();
        assert(TTx.dimensions() == TTOpA.column_dimensions());
        assert(0 <= firstIdx);
        assert(firstIdx <= lastIdx+1);
        assert(lastIdx == nDim-1);

        // calculate new entries in right_Ax when sweeping right-to-left
        for(int iDim = lastIdx - right_Ax.size(); iDim >= firstIdx; iDim--)
        {
          const auto &subTx = TTx.subTensor(iDim);
          const auto &subTOpA = TTOpA.tensorTrain().subTensor(iDim);
          Tensor3<T> subTAx;
          internal::apply_contract(TTOpA, iDim, subTOpA, subTx, subTAx);
          right_Ax.emplace_back(std::move(subTAx));
        }

        // discard old entries in right_Ax when sweeping left-to-right
        for(int iDim = lastIdx - right_Ax.size(); iDim+1 < firstIdx; iDim++)
          right_Ax.pop_back();
      }

      //! calculate next part of Ax from left to right or discard last part
      template<typename T>
      void update_left_Ax(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx, int firstIdx, int lastIdx, std::vector<Tensor3<T>>& left_Ax)
      {
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int nDim = TTx.dimensions().size();
        assert(TTx.dimensions() == TTOpA.column_dimensions());
        assert(0 == firstIdx);
        assert(firstIdx-1 <= lastIdx);
        assert(lastIdx < nDim);

        // calculate new entries in left_Ax when sweeping left-to-right
        for(int iDim = firstIdx + left_Ax.size(); iDim <= lastIdx; iDim++)
        {
            const auto& subTOpA = TTOpA.tensorTrain().subTensor(iDim);
            const auto& subTx = TTx.subTensor(iDim);
            Tensor3<T> subTAx;
            internal::apply_contract(TTOpA, iDim, subTOpA, subTx, subTAx);
            left_Ax.emplace_back(std::move(subTAx));
        }

        // discard old entries in left_Ax when sweeping right-to-left
        for(int iDim = firstIdx + left_Ax.size(); iDim-1 > lastIdx; iDim--)
          left_Ax.pop_back();
      }

      //! helper class for wrapping either a TensorTrain or just the right-most part of its sub-tensors
      template<typename T>
      class RightPartialTT final
      {
      public:
        RightPartialTT() = default;
        RightPartialTT(const TensorTrain<T>& tt) : tt_(&tt) {}
        RightPartialTT(const std::vector<Tensor3<T>>& subTs) : subTs_(&subTs) {}

        const Tensor3<T>& subTensorFromRight(int i) const
        {
          assert( tt_ || subTs_ );
          if( tt_ )
            return tt_->subTensor(tt_->dimensions().size() - 1 - i);

          return subTs_->at(i);
        }
      private:
        const TensorTrain<T> *tt_ = nullptr;
        const std::vector<Tensor3<T>> *subTs_ = nullptr;
      };

      //! helper class for wrapping either a TensorTrain or just the left-most part of its sub-tensors
      template<typename T>
      class LeftPartialTT final
      {
      public:
        LeftPartialTT() = default;
        LeftPartialTT(const TensorTrain<T>& tt) : tt_(&tt) {}
        LeftPartialTT(const std::vector<Tensor3<T>>& subTs) : subTs_(&subTs) {}

        const Tensor3<T>& subTensorFromLeft(int i) const
        {
          assert( tt_ || subTs_ );
          if( tt_ )
            return tt_->subTensor(i);
          else
            return subTs_->at(i);
        }
      private:
        const TensorTrain<T> *tt_ = nullptr;
        const std::vector<Tensor3<T>> *subTs_ = nullptr;
      };

      //! set up TT operator for the projection (assumes given TTx is correctly orthogonalized)
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

#ifndef NDEBUG
      //! helper function: return TensorTrain with additional dimension instead of boundary rank
      template<typename T>
      TensorTrain<T> removeBoundaryRank(const TensorTrain<T>& tt)
      {
        const int nDim = tt.dimensions().size();
        std::vector<Tensor3<T>> subTensors(nDim);
        for(int iDim = 0; iDim < nDim; iDim++)
          copy(tt.subTensor(iDim), subTensors[iDim]);
        const int rleft = subTensors.front().r1();
        const int rright = subTensors.back().r2();
        if( rleft != 1 )
        {
          // generate identity
          Tensor3<T> subT(1, rleft, rleft);
          subT.setConstant(T(0));
          for(int i = 0; i < rleft; i++)
            subT(0, i, i) = T(1);
          subTensors.insert(subTensors.begin(), std::move(subT));
        }
        if( rright != 1 )
        {
          // generate identity again
          Tensor3<T> subT(rright, rright, 1);
          subT.setConstant(T(0));
          for(int i = 0; i < rright; i++)
            subT(i, i, 0) = T(1);
          subTensors.insert(subTensors.end(), std::move(subT));
        }

        TensorTrain<T> extendedTT(std::move(subTensors));
        return extendedTT;
      }

      //! helper function: return TensorTrainTrain without additional empty (1x1) dimension
      template<typename T>
      TensorTrain<T> removeBoundaryRankOne(const TensorTrainOperator<T>& ttOp)
      {
        std::vector<int> rowDims = ttOp.row_dimensions();
        std::vector<int> colDims = ttOp.column_dimensions();
        std::vector<Tensor3<T>> subTensors(rowDims.size());
        for(int iDim = 0; iDim < rowDims.size(); iDim++)
          copy(ttOp.tensorTrain().subTensor(iDim), subTensors[iDim]);
        
        if( rowDims.front() == 1 && colDims.front() == 1 && subTensors.front()(0,0,0) == T(1) )
        {
          rowDims.erase(rowDims.begin());
          colDims.erase(colDims.begin());
          subTensors.erase(subTensors.begin());
        }
        if( rowDims.back() == 1 && colDims.back() == 1 && subTensors.back()(0,0,0) == T(1) )
        {
          rowDims.pop_back();
          colDims.pop_back();
          subTensors.pop_back();
        }

        TensorTrain<T> ttOp_(std::move(subTensors));

        return ttOp_;
      }
#endif

      //! calculate next part of v^Tw from right to left or discard last part
      //!
      //! Like TT dot product fused with TT apply but allows to store all intermediate results.
      //!
      //! we have
      //!  |         |
      //!  -- vTw --
      //!
      //! and we need for the next step
      //!   |               |
      //!  v_k^T --------- w_k
      //!   |              |
      //!   ----- vTw -----
      //!
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

      //! calculate next part of v^Tw from left to right or discard last part
      //!
      //! Like TT dot product fused with TT apply but allows to store all intermediate results.
      //!
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

      //! calculate the local RHS tensor-train for (M)ALS
      template<typename T>
      TensorTrain<T> calculate_local_rhs(int iDim, int nMALS, const Tensor2<T>& left_vTb, const TensorTrain<T>& TTb, const Tensor2<T>& right_vTb)
      {
        std::vector<Tensor3<T>> subT_b(nMALS);
        for(int i = 0; i < nMALS; i++)
          copy(TTb.subTensor(iDim+i), subT_b[i]);

        // first contract: tt_b_right(:,:,*) * right_vTb(:,*)
        Tensor3<T> t3_tmp;
        std::swap(subT_b.back(), t3_tmp);
        internal::dot_contract1(t3_tmp, right_vTb, subT_b.back());

        // then contract: left_vTb(*,:) * tt_b_left(*,:,:)
        std::swap(subT_b.front(), t3_tmp);
        internal::reverse_dot_contract1(left_vTb, t3_tmp, subT_b.front());

        TensorTrain<T> tt_b(std::move(subT_b));

        return tt_b;
      }

      //! calculate the local initial solutiuon in TT format for (M)ALS
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

      //! calculate the local linear operator in TT format for (M)ALS
      template<typename T>
      TensorTrainOperator<T> calculate_local_op(int iDim, int nMALS, const Tensor2<T>& left_vTAx, const TensorTrainOperator<T>& TTOp, const Tensor2<T>& right_vTAx)
      {
        std::vector<Tensor3<T>> subT_localOp(nMALS+2);
        for(int i = 0; i < nMALS; i++)
          copy(TTOp.tensorTrain().subTensor(iDim+i), subT_localOp[1+i]);
        copy_op_left(left_vTAx, subT_localOp[1].r1(), subT_localOp[0]);
        copy_op_right(right_vTAx, subT_localOp[nMALS].r2(), subT_localOp[nMALS+1]);

        std::vector<int> localRowDims(nMALS+2), localColDims(nMALS+2);
        localRowDims.front() = left_vTAx.r2();
        localColDims.front() = subT_localOp[0].n() / localRowDims.front();
        for(int i = 0; i < nMALS; i++)
        {
          localRowDims[i+1] = TTOp.row_dimensions()[iDim+i];
          localColDims[i+1] = TTOp.column_dimensions()[iDim+i];
        }
        localRowDims.back() = right_vTAx.r1();
        localColDims.back()  = subT_localOp[nMALS+1].n() / localRowDims.back();

        TensorTrainOperator<T> localTTOp(localRowDims, localColDims);
        localTTOp.tensorTrain().setSubTensors(0, std::move(subT_localOp));

        return localTTOp;
      }

      template<typename T>
      T solveDenseGMRES(const TensorTrainOperator<T>& tt_OpA, bool symmetric, const TensorTrain<T>& tt_b, TensorTrain<T>& tt_x,
                        int maxRank, int maxIter, T absTol, T relTol, const std::string& outputPrefix = "", bool verbose = false)
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
        TensorTrain<T> new_tt_x = fromDense(mv_x, mv_rhs, tt_x.dimensions(), relTol/nDim, maxRank, false, r_left, r_right);
        std::swap(tt_x, new_tt_x);

        return localRes(0);
      }

      //! helper function to returned an std::vector with the reverse ordering...
      template<typename T>
      std::vector<T> reverse(std::vector<T>&& v)
      {
        for(int i = 0; i < v.size()/2; i++)
          std::swap(v[i],v[v.size()-i-1]);
        return std::move(v);
      }

#ifndef NDEBUG
      template<typename T>
      Tensor3<T> operator-(const Tensor3<T>& a, const Tensor3<T>& b)
      {
        assert(a.r1() == b.r1());
        assert(a.n() == b.n());
        assert(a.r2() == b.r2());
        Tensor3<T> c(a.r1(), a.n(), a.r2());
        for(int i = 0; i < a.r1(); i++)
          for (int j = 0; j < a.n(); j++)
            for (int k = 0; k < a.r2(); k++)
              c(i,j,k) = a(i,j,k) - b(i,j,k);
        return c;
      }
#endif
    }
  }

  // implement TT MALS solver
  template<typename T>
  T solveMALS(const TensorTrainOperator<T>& TTOpA,
              bool symmetric,
              const MALS_projection projection,
              const TensorTrain<T>& TTb,
              TensorTrain<T>& TTx,
              int nSweeps,
              T residualTolerance,
              int maxRank,
              int nMALS, int nOverlap,
              bool useTTgmres, int gmresMaxIter, T gmresRelTol)
  {
    using namespace internal::solve_mals;
#ifndef NDEBUG
    using namespace PITTS::debug;
#endif

    // for the non-symmetric case, we can solve the normal equations, so calculate A^T*b and A^T*A
    if( projection == MALS_projection::NormalEquations )
    {
      if( symmetric )
        std::cout << "TensorTrain solveMALS: Warning - using NormalEquations variant for a symmetric operator!\n";
      TensorTrain<T> TTAtb(TTOpA.column_dimensions());
      TensorTrainOperator<T> TTOpAtA(TTOpA.column_dimensions(), TTOpA.column_dimensions());
      applyT(TTOpA, TTb, TTAtb);
      applyT(TTOpA, TTOpA, TTOpAtA);

      return solveMALS(TTOpAtA, true, MALS_projection::RitzGalerkin, TTAtb, TTx, nSweeps, residualTolerance, maxRank, nMALS, nOverlap, useTTgmres, gmresMaxIter, gmresRelTol);
    }

    if( symmetric && projection == MALS_projection::PetrovGalerkin )
    {
      std::cout << "TensorTrain solveMALS: Warning - using PetrovGalerkin projection for a symmetric operator!\n";
      // set symmetric to false because the sub-problem will become non-symmetric!
      symmetric = false;
    }

    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // check that dimensions match
    if( TTb.dimensions() != TTOpA.row_dimensions() )
      throw std::invalid_argument("TensorTrain solveMALS: operator and rhs dimensions mismatch!");
    if( TTx.dimensions() != TTOpA.column_dimensions() )
      throw std::invalid_argument("TensorTrain solveMALS: operator and x dimensions mismatch!");
    if( TTOpA.row_dimensions() != TTOpA.column_dimensions() && projection == MALS_projection::RitzGalerkin )
      throw std::invalid_argument("TensorTrain solveMALS: rectangular operator not supported with RitzGalerkin approach (row_dims != col_dims)!");


    // Generate index for sweeping (helper class)
    const int nDim = TTx.dimensions().size();
    nMALS = std::min(nMALS, nDim);
    nOverlap = std::min(nOverlap, nMALS-1);
    internal::SweepIndex lastSwpIdx(nDim, nMALS, nOverlap, -1);

    const T nrm_TTb = norm2(TTb);

#ifndef NDEBUG
    const auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());
#endif

    // check that 'symmetric' flag is used correctly
    assert(!symmetric || norm2((TTOpA-transpose(TTOpA)).tensorTrain()) < sqrt_eps);

    // we store previous parts of w^Tb from left and right
    // (respectively x^T A^T b for the non-symmetric case)
    std::vector<Tensor2<T>> left_vTb, right_vTb;
    
    // we store previous parts of x^T A x
    // (respectively x^T A^T A x for the non-symmetric case)
    std::vector<Tensor2<T>> left_vTAx, right_vTAx;

    // this includes a calculation of Ax, so allow to store the new parts of Ax in a seperate vector
    std::vector<Tensor3<T>> left_Ax, right_Ax;
    TensorTrain<T> TTAx(TTOpA.row_dimensions());

    // for the Petrov-Galerkin variant:
    // sub-tensors to represent projection space v that approximately spans Ax
    std::vector<Tensor3<T>> left_v_subT, right_v_subT;

    // calculate the error norm
    apply(TTOpA, TTx, TTAx);
    T residualNorm = axpby(T(1), TTb, T(-1), TTAx);
    std::cout << "Initial residual norm: " << residualNorm << " (abs), " << residualNorm / nrm_TTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";

    // lambda to avoid code duplication: performs one step in a sweep
    const auto solveLocalProblem = [&](const internal::SweepIndex &swpIdx, bool firstSweep = false)
    {
      std::cout << " (M)ALS setup local problem for sub-tensors " << swpIdx.leftDim() << " to " << swpIdx.rightDim() << "\n";

      internal::ensureLeftOrtho_range(TTx, 0, swpIdx.leftDim());
      update_left_Ax(TTOpA, TTx, 0, swpIdx.leftDim() - 1, left_Ax);

      internal::ensureRightOrtho_range(TTx, swpIdx.rightDim(), nDim - 1);
      update_right_Ax(TTOpA, TTx, swpIdx.rightDim() + 1, nDim - 1, right_Ax);

#ifndef NDEBUG
if( projection == MALS_projection::PetrovGalerkin )
{
  TensorTrain<T> Ax_ref = TTOpA * TTx;
  assert(left_Ax.size() == swpIdx.leftDim());
  for(int i = 0; i < left_Ax.size(); i++)
    assert(internal::t3_nrm(Ax_ref.subTensor(i) - left_Ax[i]) < sqrt_eps);
  assert(right_Ax.size() == nDim - swpIdx.rightDim() - 1);
  for(int i = 0; i < right_Ax.size(); i++)
    assert(internal::t3_nrm(Ax_ref.subTensor(nDim-i-1) - right_Ax[i]) < sqrt_eps);
}
#endif

      LeftPartialTT<T> left_v;
      RightPartialTT<T> right_v;
      // dummy tensortrain, filled later for Petrov-Galerkin case
      TensorTrain<T> TTw(0,0);

      Tensor3<T> tmp_last_left_v, tmp_last_right_v;
      if( projection == MALS_projection::RitzGalerkin )
      {
        left_v = TTx;
        right_v = TTx;
      }
      else // projection == MALS_projection::PetrovGalerkin
      {
        // stupid implementation for testing
        TensorTrainOperator<T> TTv = setupProjectionOperator(TTx, swpIdx);
        TensorTrainOperator<T> TTAv(TTOpA.row_dimensions(), TTv.column_dimensions());
        apply(TTOpA, TTv, TTAv);

#ifndef NDEBUG
{
  // check orthogonality
  TensorTrainOperator<T> TTOpI(TTv.column_dimensions(), TTv.column_dimensions());
  TTOpI.setEye();
  const TensorTrainOperator<T> vTv = transpose(TTv) * TTv;
  const T vTv_err_norm = norm2((vTv - TTOpI).tensorTrain());
  if( nMALS == 1 )
  {
    if( vTv_err_norm >= 1000*sqrt_eps )
      std::cout << "vTv-I err: " << vTv_err_norm << std::endl;
    assert(vTv_err_norm < 1000*sqrt_eps);
  }
  else
  {
    assert(vTv_err_norm < sqrt_eps);
  }
  // check A V x_local = A x
  TensorTrain<T> tt_x = calculate_local_x(swpIdx.leftDim(), nMALS, TTx);
  TensorTrain<T> TTXlocal(TTv.column_dimensions());
  TTXlocal.setOnes();
  const int leftOffset = ( tt_x.subTensor(0).r1() > 1 ) ? -1 :  0;
  TTXlocal.setSubTensors(swpIdx.leftDim() + leftOffset, removeBoundaryRank(tt_x));
  assert(norm2( TTOpA * TTx - TTAv * TTXlocal ) < sqrt_eps);
}
#endif
        TTw = calculatePetrovGalerkinProjection(TTAv, swpIdx, TTx, true);
        left_v = TTw;
        right_v = TTw;

#ifndef NDEBUG
{
  // check orthogonality
  TensorTrainOperator<T> TTOpW = setupProjectionOperator(TTw, swpIdx);
  TensorTrainOperator<T> TTOpI(TTOpW.column_dimensions(), TTOpW.column_dimensions());
  TTOpI.setEye();
  const TensorTrainOperator<T> WtW = transpose(TTOpW) * TTOpW;
  const TensorTrainOperator<T> WtW_err = WtW - TTOpI;
  //const Tensor2<T> WtW_err_dense = toDense(WtW_err);
  //std::cout << "WtW_err:\n" << ConstEigenMap(WtW_err_dense) << std::endl;
  assert(norm2((transpose(TTOpW) * TTOpW - TTOpI).tensorTrain()) < sqrt_eps);
}
#endif
      }

      update_left_vTw<T>(left_v, TTb, 0, swpIdx.leftDim() - 1, left_vTb);
      update_left_vTw<T>(left_v, left_Ax, 0, swpIdx.leftDim() - 1, left_vTAx);

      update_right_vTw<T>(right_v, TTb, swpIdx.rightDim() + 1, nDim - 1, right_vTb);
      update_right_vTw<T>(right_v, right_Ax, swpIdx.rightDim() + 1, nDim - 1, right_vTAx);


      // prepare operator and right-hand side
      TensorTrain<T> tt_x = calculate_local_x(swpIdx.leftDim(), nMALS, TTx);
      const TensorTrain<T> tt_b = calculate_local_rhs(swpIdx.leftDim(), nMALS, left_vTb.back(), TTb, right_vTb.back());
      const TensorTrainOperator<T> localTTOp = calculate_local_op(swpIdx.leftDim(), nMALS, left_vTAx.back(), TTOpA, right_vTAx.back());
#ifndef NDEBUG
{
  // check that localTTOp * tt_x is valid
  assert(localTTOp.column_dimensions()[0] == tt_x.subTensor(0).r1());
  for(int i = 0; i < nMALS; i++)
    assert(localTTOp.column_dimensions()[i+1] == tt_x.dimensions()[i]);
  assert(localTTOp.column_dimensions()[nMALS+1] == tt_x.subTensor(nMALS-1).r2());
  // check that localTTOp^T * tt_b is valid
  assert(localTTOp.row_dimensions()[0] == tt_b.subTensor(0).r1());
  for(int i = 0; i < nMALS; i++)
    assert(localTTOp.row_dimensions()[i+1] == tt_b.dimensions()[i]);
  assert(localTTOp.row_dimensions()[nMALS+1] == tt_b.subTensor(nMALS-1).r2());
}
#endif
      if( projection == MALS_projection::RitzGalerkin )
      {
        assert(std::abs(dot(tt_x, tt_b) - dot(TTx, TTb)) <= sqrt_eps*std::abs(dot(TTx, TTb)));
      }
      else if( projection == MALS_projection::PetrovGalerkin )
      {
#ifndef NDEBUG
{
  TensorTrainOperator<T> TTOpV = setupProjectionOperator(TTx, swpIdx);
  TensorTrainOperator<T> TTOpW = setupProjectionOperator(TTw, swpIdx);
  // check W^T b = tt_b
  TensorTrain<T> TTblocal(TTOpW.column_dimensions());
  TTblocal.setOnes();
  int leftOffset = ( tt_b.subTensor(0).r1() > 1 ) ? -1 :  0;
  TTblocal.setSubTensors(swpIdx.leftDim() + leftOffset, removeBoundaryRank(tt_b));
  assert(norm2( transpose(TTOpW) * TTb -  TTblocal ) < sqrt_eps);

  // check localTTOp = W^T A V
  TensorTrainOperator<T> WtAV_ref = transpose(TTOpW) * TTOpA * TTOpV;
  TensorTrainOperator<T> WtAV(TTOpW.column_dimensions(), TTOpV.column_dimensions());
  WtAV.setOnes();
  leftOffset = ( localTTOp.tensorTrain().subTensor(0).n() == 1 && localTTOp.tensorTrain().subTensor(0)(0,0,0) == T(1) ) ? 0 : -1;
  WtAV.tensorTrain().setSubTensors(swpIdx.leftDim() + leftOffset, removeBoundaryRankOne(localTTOp));
  assert(norm2( WtAV.tensorTrain() - WtAV_ref.tensorTrain() ) < sqrt_eps*norm2(WtAV_ref.tensorTrain()));
}
#endif
      }
      // first Sweep: let GMRES start from zero, at least favorable for TT-GMRES!
      if (firstSweep && residualNorm / nrm_TTb > 0.5)
        tt_x.setZero();
      
      if (useTTgmres)
        const T localRes = solveGMRES(localTTOp, tt_b, tt_x, gmresMaxIter, gmresRelTol * residualTolerance * nrm_TTb, gmresRelTol, maxRank, true, symmetric, " (M)ALS local problem: ", true);
      else
        const T localRes = solveDenseGMRES(localTTOp, symmetric, tt_b, tt_x, maxRank, gmresMaxIter, gmresRelTol * residualTolerance * nrm_TTb, gmresRelTol, " (M)ALS local problem: ", true);

      TTx.setSubTensors(swpIdx.leftDim(), std::move(tt_x));

      if( projection == MALS_projection::PetrovGalerkin )
      {
        // recover original sub-tensor of projection space
        {
          // this also invalidates left_vTb and left_vTAx!
          left_vTb.clear();
          left_vTAx.clear();
        }
        {
          // this also invalidates right_vTb and right_vTAx!
          right_vTb.clear();
          right_vTAx.clear();
        }
      }
    };

    // AMEn idea: enhance subspace for ALS with some directions of the global residual
    const auto enhanceSubSpace = [&](const internal::SweepIndex &swpIdx, bool leftToRight, int maxAdditionalRank)
    {
      const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();
      if( leftToRight && swpIdx.rightDim() == swpIdx.nDim()-1 )
        return;
      if( (!leftToRight) && swpIdx.leftDim() == 0 )
        return;
      
      // only intended for nMALS == 1
      assert(swpIdx.leftDim() == swpIdx.rightDim());
      TensorTrain<T> TTr(TTb.dimensions());
      {
        const int iDim = leftToRight ? swpIdx.rightDim() + 1 : swpIdx.leftDim() - 1;

        // also needed in solveLocalProblem but should not be extra work if called twice
        internal::ensureLeftOrtho_range(TTx, 0, iDim);
        internal::ensureRightOrtho_range(TTx, iDim, nDim - 1);
        update_left_Ax(TTOpA, TTx, 0, iDim, left_Ax);
        update_right_Ax(TTOpA, TTx, iDim + 1, nDim - 1, right_Ax);

        std::vector<Tensor3<T>> subT_Ax(nDim);
        for(int i = 0; i < left_Ax.size(); i++)
          copy(left_Ax[i], subT_Ax[i]);
        left_Ax.pop_back();
        for(int i = 0; i < right_Ax.size(); i++)
          copy(right_Ax[i], subT_Ax[nDim-i-1]);
        subT_Ax = TTr.setSubTensors(0, std::move(subT_Ax));
        axpby(T(1), TTb, T(-1), TTr, T(0));
        //T rTx = dot(TTr, TTx);
        //axpby(rTx, TTx, T(-1), TTr, T(0));
        internal::ensureLeftOrtho_range(TTr, 0, iDim);
        internal::ensureRightOrtho_range(TTr, iDim, nDim-1);
      }

      if( leftToRight )
      {
        std::vector<Tensor3<T>> newSubT(2);
        {
          const Tensor3<T>& subT0 = TTx.subTensor(swpIdx.rightDim());
          const Tensor3<T>& subTr = TTr.subTensor(swpIdx.rightDim());
          const Tensor3<T>& subT1 = TTx.subTensor(swpIdx.rightDim()+1);
          const int addRank = std::min<int>(maxAdditionalRank, subTr.r2());
          std::cout << " Enhancing subspace (left-to-right) for sub-tensor " << swpIdx.rightDim() << " for optimizing sub-tensor " << swpIdx.rightDim()+1 << ": increasing rank from " << subT0.r2() << " to " << subT0.r2()+addRank << "\n";
          //MultiVector<T> mv0, mvr;
          //unfold_left(subT0, mv0);
          //unfold_left(subTr, mvr);
          //std::cout << "subT0:\n" << ConstEigenMap(mv0) << "\n";
          //std::cout << "subTr:\n" << ConstEigenMap(mvr) << "\n";
          newSubT[0].resize(subT0.r1(), subT0.n(), subT0.r2()+addRank);
          newSubT[1].resize(subT1.r1()+addRank, subT1.n(), subT1.r2());
          newSubT[0].setConstant(T(0));
          newSubT[1].setConstant(T(0));
          for(int i = 0; i < subT0.r1(); i++)
            for(int j = 0; j < subT0.n(); j++)
              for(int k = 0; k < subT0.r2(); k++)
                newSubT[0](i,j,k) = subT0(i,j,k);
          for(int i = 0; i < std::min(subT0.r1(), subTr.r1()); i++)
            for(int j = 0; j < subT0.n(); j++)
              for(int k = 0; k < addRank; k++)
                newSubT[0](i,j,subT0.r2()+k) = subTr(i,j,k);
          for(int i = 0; i < subT1.r1(); i++)
            for(int j = 0; j < subT1.n(); j++)
              for(int k = 0; k < subT1.r2(); k++)
                newSubT[1](i,j,k) = subT1(i,j,k);
        }
        TTx.setSubTensors(swpIdx.rightDim(), std::move(newSubT));

        // these are not valid any more
        left_Ax.pop_back();
        if( !left_vTAx.empty() )
          left_vTAx.pop_back();
        if( !left_vTb.empty() )
          left_vTb.pop_back();
      }
      else // right-to-left
      {

        std::vector<Tensor3<T>> newSubT(2);
        {
          const Tensor3<T>& subT0 = TTx.subTensor(swpIdx.leftDim()-1);
          const Tensor3<T>& subTr = TTr.subTensor(swpIdx.leftDim());
          const Tensor3<T>& subT1 = TTx.subTensor(swpIdx.leftDim());
          const int addRank = std::min<int>(maxAdditionalRank, subTr.r1());
          std::cout << " Enhancing subspace (right-to-left) for sub-tensor " << swpIdx.leftDim() << " for optimizing sub-tensor " << swpIdx.leftDim()-1 << ": increasing rank from " << subT0.r2() << " to " << subT0.r2()+addRank << "\n";
          newSubT[0].resize(subT0.r1(), subT0.n(), subT0.r2()+addRank);
          newSubT[1].resize(subT1.r1()+addRank, subT1.n(), subT1.r2());
          newSubT[0].setConstant(T(0));
          newSubT[1].setConstant(T(0));
          for(int i = 0; i < subT0.r1(); i++)
            for(int j = 0; j < subT0.n(); j++)
              for(int k = 0; k < subT0.r2(); k++)
                newSubT[0](i,j,k) = subT0(i,j,k);
          for(int i = 0; i < subT1.r1(); i++)
            for(int j = 0; j < subT1.n(); j++)
              for(int k = 0; k < subT1.r2(); k++)
                newSubT[1](i,j,k) = subT1(i,j,k);
          for(int i = 0; i < addRank; i++)
            for(int j = 0; j < subT0.n(); j++)
              for(int k = 0; k < std::min(subT1.r2(), subTr.r2()); k++)
                newSubT[1](subT1.r1()+i,j,k) = subTr(i,j,k);
        }
        TTx.setSubTensors(swpIdx.leftDim()-1, std::move(newSubT));

        // these are not valid any more
        right_Ax.pop_back();
        if( !right_vTAx.empty() )
          right_vTAx.pop_back();
        if( !right_vTb.empty() )
          right_vTb.pop_back();
       }
    };

    // now everything is prepared, perform the sweeps
    for(int iSweep = 0; iSweep < nSweeps; iSweep++)
    {
      if( residualNorm / nrm_TTb < residualTolerance )
        break;

      // sweep left to right
      for(auto swpIdx = lastSwpIdx.first(); swpIdx; swpIdx = swpIdx.next())
      {
        // skip iteration if this is the same as in the last right-to-left sweep
        if( nMALS != nDim && swpIdx == lastSwpIdx )
          continue;

        if( nMALS == 1 && swpIdx.leftDim() > 0 )
          enhanceSubSpace(swpIdx.previous(), true, 2);

        solveLocalProblem(swpIdx, iSweep == 0);
        
        /* // not needed/useful: truncate again
        if( nMALS == 1 && swpIdx.leftDim() > 0 )
        {
          internal::ensureRightOrtho_range(TTx, swpIdx.leftDim()-1, nDim-1);
          internal::leftNormalize_range(TTx, swpIdx.leftDim()-1, swpIdx.leftDim(), gmresRelTol*residualTolerance);
          right_vTb.clear();
          right_vTAx.clear();
          right_Ax.clear();
          left_vTb.clear();
          left_vTAx.clear();
          left_Ax.clear();
        }
        */

        lastSwpIdx = swpIdx;
      }
      // update remaining sub-tensors of left_Ax
      update_left_Ax(TTOpA, TTx, 0, nDim - 1, left_Ax);
      left_Ax = TTAx.setSubTensors(0, std::move(left_Ax));
      // for non-symm. cases, we still need left_Ax
      if( projection == MALS_projection::PetrovGalerkin || nMALS == 1 )
        for(int iDim = 0; iDim < nDim; iDim++)
          copy(TTAx.subTensor(iDim), left_Ax[iDim]);



      assert( norm2(TTOpA * TTx - TTAx) < sqrt_eps );

      // check error
      residualNorm = axpby(T(1), TTb, T(-1), TTAx);
      std::cout << "Sweep " << iSweep+0.5 << " residual norm: " << residualNorm << " (abs), " << residualNorm / nrm_TTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";
      if( residualNorm / nrm_TTb < residualTolerance )
        break;

      // sweep right to left
      for(auto swpIdx = lastSwpIdx.last(); swpIdx; swpIdx = swpIdx.previous())
      {
        // skip iteration if this is the same as in the last left-to-right sweep
        if( nMALS != nDim && swpIdx == lastSwpIdx )
          continue;

        //if( nMALS == 1 && swpIdx.rightDim() < nDim-1 )
        //  enhanceSubSpace(swpIdx.next(), false, 4);

        solveLocalProblem(swpIdx);
        lastSwpIdx = swpIdx;
      }
      // update remaining sub-tensors of right_Ax
      update_right_Ax(TTOpA, TTx, 0, nDim - 1, right_Ax);
      // TODO: that's wrong -> need a reverse
      right_Ax = TTAx.setSubTensors(0, reverse(std::move(right_Ax)));
      // for non-symm. cases, we still need right_Ax
      if( projection == MALS_projection::PetrovGalerkin || nMALS == 1 )
        for(int iDim = 0; iDim < nDim; iDim++)
          copy(TTAx.subTensor(iDim), right_Ax[nDim-iDim-1]);

      assert(norm2(TTOpA * TTx - TTAx) < sqrt_eps);

      // check error
      residualNorm = axpby(T(1), TTb, T(-1), TTAx);
      std::cout << "Sweep " << iSweep+1 << " residual norm: " << residualNorm << " (abs), " << residualNorm / nrm_TTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";
    }


    return residualNorm;
  }

}


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_IMPL_HPP
