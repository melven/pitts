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
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensor3_fold.hpp"
#include "pitts_tensor3_unfold.hpp"
#include "pitts_eigen.hpp"
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
        using ResultType = std::pair<Tensor3<T>,Tensor2<T>>;
        return [&](int iDim, optional_cref<ResultType> prev_QB, ResultType& QB)
        {
          const auto& subTz = TTz.subTensor(iDim);

          Tensor2<T> t2;
          if( !prev_QB )
          {
            unfold_left(subTz, t2);
          }
          else
          {
            // contract prev_t2(:,*) * subTz(*,:,:)
            internal::normalize_contract1(prev_QB->get().second, subTz, QB.first);
            unfold_left(QB.first, t2);
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
        using ResultType = std::pair<Tensor3<T>,Tensor2<T>>;
        return [&](int iDim, optional_cref<ResultType> prev_QB, ResultType& QB)
        {
          const auto& subTz = TTz.subTensor(iDim);

          Tensor2<T> t2;
          if( !prev_QB )
          {
            unfold_right(subTz, t2);
          }
          else
          {
            // contract subTz(:,:,*) * prev_t2(*,:)
            internal::normalize_contract2(subTz, prev_QB->get().second, QB.first);
            unfold_right(QB.first, t2);
          }

          // calculate LQt of subT(: : x :)
          auto [B, Qt] = internal::normalize_qb(t2, false);
          fold_right(Qt, subTz.n(), QB.first);
          QB.second = std::move(B);
        };
      }

      template<typename T>
      constexpr auto axpby_loop_from_left(const auto& TTx, const auto& TTy)
      {
        using ResultType = std::pair<Tensor3<T>,Tensor2<T>>;
        return [&](int iDim, optional_cref<ResultType> prev_QB, ResultType& QB)
        {
          const auto& subTx = TTx.subTensor(iDim).first;
          const auto& subTy = TTy.subTensor(iDim);
          assert(subTx.n() == subTy.n());
          const int n = subTx.n();
          const int r2x = subTx.r2();
          const int r2y = subTy.r2();

          Tensor2<T> t2;
          using mat = Eigen::MatrixX<T>;
          if( !prev_QB )
          {
            assert(subTx.r1() == 1 && subTy.r1() == 1);
            Eigen::Map<const mat> mapX(&subTx(0,0,0), n, r2x);
            Eigen::Map<const mat> mapY(&subTy(0,0,0), n, r2y);
            t2.resize(n, r2x+r2y);
            EigenMap(t2).leftCols (r2x) = mapX;
            EigenMap(t2).rightCols(r2y) = mapY;
          }
          else
          {
            const int r1x = subTx.r1();
            const int r1y = subTy.r1();
            Eigen::Map<const mat> mapX(&subTx(0,0,0), r1x, n*r2x);
            Eigen::Map<const mat> mapY(&subTy(0,0,0), r1y, n*r2y);
            // contract prev_B * (X 0; 0 Y)
            const auto& prev_B = prev_QB->get().second;
            t2.resize(prev_B.r1(), n*r2x+n*r2y);
            EigenMap(t2).leftCols (n*r2x) = ConstEigenMap(prev_B).leftCols(r1x) * mapX;
            EigenMap(t2).rightCols(n*r2y) = ConstEigenMap(prev_B).rightCols(r1y) * mapY;
            t2.resize(t2.r1()*n, r2x+r2y, false);
          }

          // calculate QR of subT(: : x :)
          auto [Q, B] = internal::normalize_qb(t2, true);
          fold_left(Q, n, QB.first);
          QB.second = std::move(B);
        };
      }

      template<typename T>
      const auto& extract_first(const T& t) {return t;}

      template<typename T1, typename T2>
      const auto& extract_first(const std::pair<T1,T2>& t12) {return t12.first;}

      template<typename T>
      constexpr auto axpby_loop_from_right(const auto& TTx, const auto& TTy)
      {
        using ResultType = std::pair<Tensor3<T>,Tensor2<T>>;
        return [&](int iDim, optional_cref<ResultType> prev_QB, ResultType& QB)
        {
          const auto& subTx = extract_first(TTx.subTensor(iDim));
          const auto& subTy = extract_first(TTy.subTensor(iDim));
          assert(subTx.n() == subTy.n());
          const int n = subTx.n();
          const int r1x = subTx.r1();
          const int r1y = subTy.r1();

          Tensor2<T> t2;
          using mat = Eigen::MatrixX<T>;
          if( !prev_QB )
          {
            assert(subTx.r2() == 1 && subTy.r2() == 1);
            Eigen::Map<const mat> mapX(&subTx(0,0,0), r1x, n);
            Eigen::Map<const mat> mapY(&subTy(0,0,0), r1y, n);
            t2.resize(r1x+r1y,n);
            EigenMap(t2).topRows   (r1x) = mapX;
            EigenMap(t2).bottomRows(r1y) = mapY;
          }
          else
          {
            const int r2x = subTx.r2();
            const int r2y = subTy.r2();
            // contract (X 0; 0 Y) * prev_B
            // unfortunately, the memory layout is strided for the right-to-left axpby loop, so we need to copy stuff around...
            const auto& prev_B = prev_QB->get().second;
            auto& t3 = QB.first;
            t3.resize(r1x+r1y,n,r2x+r2y);
#pragma omp parallel for schedule(static) collapse(2) if((r2x+r2y)*n > 50)
            for(int k = 0; k < r2x+r2y; k++)
              for(int j = 0; j < n; j++)
                for(int i = 0; i < r1x+r1y; i++)
                  t3(i,j,k) = k < r2x && i < r1x ? subTx(i,j,k) : k >= r2x && i >= r1x ? subTy(i-r1x,j,k-r2x) : T(0);
            Tensor2<T> tmp;
            unfold_left(t3, tmp);
            t2.resize((r1x+r1y)*n, prev_B.r2());
            EigenMap(t2) = ConstEigenMap(tmp) * ConstEigenMap(prev_B);
            t2.resize(r1x+r1y, n*t2.r2(), false);
          }

          // calculate LQt of subT(: x : :)
          auto [B, Qt] = internal::normalize_qb(t2, false);
          fold_right(Qt, n, QB.first);
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
