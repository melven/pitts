/*! @file pitts_tensortrain_solve_mals_helper.hpp
* @brief Helper functionality for PITTS::solveMALS
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-05-05
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SOLVE_MALS_HELPER_HPP
#define PITTS_TENSORTRAIN_SOLVE_MALS_HELPER_HPP

// includes
#include "pitts_tensortrain_sweep_index.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_operator.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! dedicated helper functions for solveMALS
    namespace solve_mals
    {
      template<typename ResultType, typename IntermediateType, typename LeftToRightFunction, typename RightToLeftFunction>
      class SweepData final
      {
        public:
          SweepData(int nDim, LeftToRightFunction, RightToLeftFunction) : nDim_(nDim), result_(nDim_), intermediate_(nDim_) {}
        private:
          int nDim_;
          std::vector<ResultType> result_;
          std::vector<IntermediateType> intermediate_;
      };

      //! calculate next part of Ax from right to left or discard last part
      template<typename T>
      void update_right_Ax(const TensorTrainOperator<T> TTOpA, const TensorTrain<T>& TTx, int firstIdx, int lastIdx,
                           std::vector<Tensor3<T>>& right_Ax, std::vector<Tensor3<T>>& right_Ax_ortho, std::vector<Tensor2<T>>& right_Ax_ortho_M);


      //! calculate next part of Ax from left to right or discard last part
      template<typename T>
      void update_left_Ax(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx, int firstIdx, int lastIdx,
                          std::vector<Tensor3<T>>& left_Ax, std::vector<Tensor3<T>>& left_Ax_ortho, std::vector<Tensor2<T>>& left_Ax_ortho_M);


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
      //!
      //! TODO: use only for debugging; currently still needed for PetrovGalerkin variant
      //!
      template<typename T>
      TensorTrainOperator<T> setupProjectionOperator(const TensorTrain<T>& TTx, SweepIndex swpIdx);

      template<typename T>
      TensorTrain<T> calculatePetrovGalerkinProjection(TensorTrainOperator<T>& TTAv, SweepIndex swpIdx, const TensorTrain<T>& TTx, bool symmetrize);
      

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
      void update_right_vTw(const RightPartialTT<T>& TTv, const RightPartialTT<T>& TTw, int firstIdx, int lastIdx, std::vector<Tensor2<T>>& right_vTw);
      

      //! calculate next part of v^Tw from left to right or discard last part
      //!
      //! Like TT dot product fused with TT apply but allows to store all intermediate results.
      //!
      template<typename T>
      void update_left_vTw(const LeftPartialTT<T>& TTv, const LeftPartialTT<T>& TTw, int firstIdx, int lastIdx, std::vector<Tensor2<T>>& left_vTw);
      

      //! calculate the local RHS tensor-train for (M)ALS
      template<typename T>
      TensorTrain<T> calculate_local_rhs(int iDim, int nMALS, const Tensor2<T>& left_vTb, const TensorTrain<T>& TTb, const Tensor2<T>& right_vTb);
      

      //! calculate the local initial solutiuon in TT format for (M)ALS
      template<typename T>
      TensorTrain<T> calculate_local_x(int iDim, int nMALS, const TensorTrain<T>& TTx);
      

      //! calculate the local linear operator in TT format for (M)ALS
      template<typename T>
      TensorTrainOperator<T> calculate_local_op(int iDim, int nMALS, const Tensor2<T>& left_vTAx, const TensorTrainOperator<T>& TTOp, const Tensor2<T>& right_vTAx);
      

      template<typename T>
      T solveDenseGMRES(const TensorTrainOperator<T>& tt_OpA, bool symmetric, const TensorTrain<T>& tt_b, TensorTrain<T>& tt_x,
                        int maxRank, int maxIter, T absTol, T relTol, const std::string& outputPrefix, bool verbose);
      

      //! helper function to returned an std::vector with the reverse ordering...
      template<typename T>
      std::vector<T> reverse(std::vector<T>&& v)
      {
        for(int i = 0; i < v.size()/2; i++)
          std::swap(v[i],v[v.size()-i-1]);
        return std::move(v);
      }
    }
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_solve_mals_helper_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_SOLVE_MALS_HELPER_HPP
