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
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_unfold.hpp"
#include <optional>
#include <functional>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! dedicated helper functions for solveMALS
    namespace solve_mals
    {
      template<typename T>
      using optional_cref = std::optional<std::reference_wrapper<const T>>;

      template<typename ResultType, typename LeftToRightFunction, typename RightToLeftFunction>
      class SweepData final
      {
        public:
          SweepData(int nDim, LeftToRightFunction&& leftToRight, RightToLeftFunction rightToLeft) :
            nDim_(nDim), leftDim_(-1), rightDim_(nDim_), result_(nDim),
            updateLeftToRight_(std::move(leftToRight)),
            updateRightToLeft_(std::move(rightToLeft))
          {
          }
          [[nodiscard]] int nDim() const {return nDim_;}
          [[nodiscard]] int leftDim() const {return leftDim_;}
          [[nodiscard]] int rightDim() const {return rightDim_;}
          [[nodiscard]] const ResultType& subTensor(int iDim) const
          {
            assert((iDim >= 0 && iDim <= leftDim_) || (iDim >= rightDim_ && iDim <= nDim_-1));
            return result_[iDim];
          }
          [[nodiscard]] const std::vector<ResultType>& data() const {return result_;}
          void invalidate(int iDim, int nMALS = 1)
          {
            iDim = std::clamp(iDim, 0, nDim_-1);
            leftDim_ = std::min(leftDim_, iDim-1);
            rightDim_ = std::max(rightDim_, iDim+nMALS);
          }
          void update(int newLeftDim, int newRightDim)
          {
            assert(leftDim_ >= -1 && leftDim_ < rightDim_ && rightDim_ <= nDim_);
            assert(newLeftDim >= -1 && newLeftDim < newRightDim && newRightDim <= nDim_);

            for(; leftDim_ < newLeftDim; leftDim_++)
              updateLeftToRight_(leftDim_+1, left(), result_[leftDim_+1]);
            leftDim_ = newLeftDim;

            for(; rightDim_ > newRightDim; rightDim_--)
              updateRightToLeft_(rightDim_-1, right(), result_[rightDim_-1]);
            rightDim_ = newRightDim;
          }
          [[nodiscard]] optional_cref<ResultType> left() const
          {
            if(leftDim_ < 0)
              return std::nullopt;
            return std::ref(result_[leftDim_]);
          }
          [[nodiscard]] optional_cref<ResultType> right() const
          {
            if(rightDim_ >= nDim_) 
              return std::nullopt;
            return std::cref(result_[rightDim_]);
          }
        private:
          int nDim_;
          int leftDim_ = 0;
          int rightDim_ = nDim_-1;
          std::vector<ResultType> result_;
          LeftToRightFunction updateLeftToRight_;
          RightToLeftFunction updateRightToLeft_;
      };

      //! for template argument deduction...
      template<typename ResultType, typename LeftToRightFunction, typename RightToLeftFunction>
      SweepData<ResultType, LeftToRightFunction, RightToLeftFunction> defineSweepData(int nDim, LeftToRightFunction&& leftToRight, RightToLeftFunction rightToLeft)
      {
        return {nDim, std::forward<LeftToRightFunction>(leftToRight), std::forward<RightToLeftFunction>(rightToLeft)};
      }

      template<typename T>
      constexpr auto apply_loop(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx)
      {
        return [&](int iDim, optional_cref<Tensor3<T>>, Tensor3<T>& subTAx)
        {
          const auto& subTOpA = TTOpA.tensorTrain().subTensor(iDim);
          const auto& subTx = TTx.subTensor(iDim);
          internal::apply_contract(TTOpA, iDim, subTOpA, subTx, subTAx);
        };
      }

      template<typename T>
      constexpr auto dot_loop_from_left(const auto& TTv, const auto& TTw)
      {
        return [&](int iDim, optional_cref<Tensor2<T>> prev_t2, Tensor2<T>& t2)
        {
          const auto& subTv = TTv.subTensor(iDim);
          const auto& subTw = TTw.subTensor(iDim);
          if( !prev_t2 )
          {
            // contract: subTw(*,*,:) * subTv(*,*,:)
            internal::reverse_dot_contract2(subTw, subTv, t2);
          }
          else
          {
            Tensor3<T> t3_tmp;
            // first contraction: prev_t2(*,:) * subTw(*,:,:)
            internal::reverse_dot_contract1<T>(*prev_t2, subTw, t3_tmp);
            // second contraction: t3(*,*,:) * subTv(*,*,:)
            internal::reverse_dot_contract2(t3_tmp, subTv, t2);
          }
        };
      }

      template<typename T>
      constexpr auto dot_loop_from_right(const auto& TTv, const auto& TTw)
      {
        return [&](int iDim, optional_cref<Tensor2<T>> prev_t2, Tensor2<T>& t2)
        {
          const auto& subTv = TTv.subTensor(iDim);
          const auto& subTw = TTw.subTensor(iDim);
          if( !prev_t2 )
          {
            // contract: subTw(:,*,*) * subTv(:,*,*)
            internal::dot_contract2(subTv, subTw, t2);
          }
          else
          {
            Tensor3<T> t3_tmp;
            // first contraction: subTw(:,:,*) * prev_t2(:,*)
            internal::dot_contract1<T>(subTw, *prev_t2, t3_tmp);
            // second contraction: subTv(:,*,*) * t3_tmp(:,*,*)
            internal::dot_contract2(subTv, t3_tmp, t2);
          }
        };
      }

      template<typename T>
      constexpr auto ortho_loop_from_left(const auto& TTz)
      {
        return [&](int iDim, optional_cref<std::pair<Tensor3<T>,Tensor2<T>>> prev_QB, std::pair<Tensor3<T>,Tensor2<T>>& QB)
        {
          const auto& subTz = TTz.subTensor(iDim);

          Tensor2<T> t2;
          if( prev_QB )
          {
            // now contract prev_t2(:,*) * subTz(*,:,:)
            internal::normalize_contract1(prev_QB->get().second, subTz, QB.first);
            unfold_left(QB.first, t2);
          }
          else
          {
            unfold_left(subTz, t2);
          }

          // calculate QR of subT(: : x :)
          auto [Q, B] = internal::normalize_qb(t2, true);
          fold_left(Q, subTz.n(), QB.first);
          QB.second = std::move(B);
        };
      }

      template<typename T>
      constexpr auto ortho_loop_from_right(const auto& TTz)
      {
        return [&](int iDim, optional_cref<std::pair<Tensor3<T>,Tensor2<T>>> prev_QB, std::pair<Tensor3<T>,Tensor2<T>>& QB)
        {
          const auto& subTz = TTz.subTensor(iDim);

          Tensor2<T> t2;
          if( prev_QB )
          {
            // now contract subTz(:,:,*) * prev_t2(*,:)
            internal::normalize_contract2(subTz, prev_QB->get().second, QB.first);
            unfold_right(QB.first, t2);
          }
          else
          {
            unfold_right(subTz, t2);
          }

          // calculate QR of subT(: : x :)
          auto [B, Q] = internal::normalize_qb(t2, false);
          fold_right(Q, subTz.n(), QB.first);
          QB.second = std::move(B);
        };
      }

      //! set up TT operator for the projection (assumes given TTx is correctly orthogonalized)
      //!
      //! TODO: use only for debugging; currently still needed for PetrovGalerkin variant
      //!
      template<typename T>
      TensorTrainOperator<T> setupProjectionOperator(const TensorTrain<T>& TTx, SweepIndex swpIdx);

      template<typename T>
      TensorTrain<T> calculatePetrovGalerkinProjection(TensorTrainOperator<T>& TTAv, SweepIndex swpIdx, const TensorTrain<T>& TTx, bool symmetrize);
      

      //! calculate the local RHS tensor-train for (M)ALS
      template<typename T>
      TensorTrain<T> calculate_local_rhs(int iDim, int nMALS, optional_cref<Tensor2<T>> left_vTb, const TensorTrain<T>& TTb, optional_cref<Tensor2<T>> right_vTb);
      

      //! calculate the local initial solutiuon in TT format for (M)ALS
      template<typename T>
      TensorTrain<T> calculate_local_x(int iDim, int nMALS, const TensorTrain<T>& TTx);
      

      //! calculate the local linear operator in TT format for (M)ALS
      template<typename T>
      TensorTrainOperator<T> calculate_local_op(int iDim, int nMALS, optional_cref<Tensor2<T>> left_vTAx, const TensorTrainOperator<T>& TTOp, optional_cref<Tensor2<T>> right_vTAx);
      

      template<typename T>
      T solveDenseGMRES(const TensorTrainOperator<T>& tt_OpA, bool symmetric, const TensorTrain<T>& tt_b, TensorTrain<T>& tt_x,
                        int maxRank, int maxIter, T absTol, T relTol, const std::string& outputPrefix, bool verbose);
    }
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_solve_mals_helper_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_SOLVE_MALS_HELPER_HPP
