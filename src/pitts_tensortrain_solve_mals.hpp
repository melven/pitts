/*! @file pitts_tensortrain_solve_mals.hpp
* @brief MALS algorithm for solving (non-)symmetric linear systems in TT format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-28
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SOLVE_MALS_HPP
#define PITTS_TENSORTRAIN_SOLVE_MALS_HPP

// includes
//#include <omp.h>
//#include <iostream>
#include <cmath>
#include <limits>
#include <cassert>
#include <iostream>
#include <utility>
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensor3_combine.hpp"
#include "pitts_tensor3_split.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_operator_apply_dense.hpp"
#include "pitts_tensortrain_operator_apply_transposed.hpp"
#include "pitts_tensortrain_operator_apply_transposed_op.hpp"
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
          else
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
      void copy_op_left(const Tensor2<T>& t2, Tensor3<T>& t3)
      {
          const int r1 = t2.r2();
          const int rAl = t2.r1() / r1;

          const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
              {{"r1", "rAl"}, {r1, rAl}},   // arguments
              {{r1*r1*rAl*kernel_info::NoOp<T>()},    // flops
              {r1*r1*rAl*kernel_info::Store<T>() + r1*r1*rAl*kernel_info::Load<T>()}}  // data
              );

          t3.resize(1,r1*r1,rAl);

#pragma omp parallel for collapse(3) schedule(static) if(r1*r1*rAl > 500)
          for(int i = 0; i < r1; i++)
            for(int j = 0; j < r1; j++)
              for(int k = 0; k < rAl; k++)
                t3(0,i+j*r1,k) = t2(j+k*r1,i);
      }

      template<typename T>
      void copy_op_right(const Tensor2<T>& t2, Tensor3<T>& t3)
      {
          const int r2 = t2.r1();
          const int rAr = t2.r2() / r2;

          const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
              {{"r2", "rAr"}, {r2, rAr}},   // arguments
              {{r2*r2*rAr*kernel_info::NoOp<T>()},    // flops
              {r2*r2*rAr*kernel_info::Store<T>() + r2*r2*rAr*kernel_info::Load<T>()}}  // data
              );

          t3.resize(rAr,r2*r2,1);

#pragma omp parallel for collapse(3) schedule(static) if(r2*r2*rAr > 500)
          for(int i = 0; i < r2; i++)
            for(int j = 0; j < r2; j++)
              for(int k = 0; k < rAr; k++)
                t3(k,i+j*r2,0) = t2(i,j+k*r2);
      }

      //! calculate the local linear operator in TT format for (M)ALS
      template<typename T>
      TensorTrainOperator<T> calculate_local_op(int iDim, int nMALS, const Tensor2<T>& left_vTAx, const TensorTrainOperator<T>& TTOp, const Tensor2<T>& right_vTAx)
      {
        const int n0 = left_vTAx.r2();
        const int nd = right_vTAx.r1();

        std::vector<int> localRowDims(nMALS+2), localColDims(nMALS+2);
        localRowDims.front() = localColDims.front() = n0;
        for(int i = 0; i < nMALS; i++)
        {
          localRowDims[i+1] = TTOp.row_dimensions()[iDim+i];
          localColDims[i+1] = TTOp.column_dimensions()[iDim+i];
        }
        localRowDims.back()  = localColDims.back()  = nd;

        std::vector<Tensor3<T>> subT_localOp(nMALS+2);
        copy_op_left(left_vTAx, subT_localOp.front());
        for(int i = 0; i < nMALS; i++)
          copy(TTOp.tensorTrain().subTensor(iDim+i), subT_localOp[1+i]);
        copy_op_right(right_vTAx, subT_localOp.back());

        TensorTrainOperator<T> localTTOp(localRowDims, localColDims);
        localTTOp.tensorTrain().setSubTensors(0, std::move(subT_localOp));

        return localTTOp;
      }

      template<typename T>
      T solveDenseGMRES(const TensorTrainOperator<T>& tt_OpA, bool symmetric, const TensorTrain<T>& tt_b, TensorTrain<T>& tt_x,
                        int maxRank, int maxIter, T absTol, T relTol, const std::string& outputPrefix = "", bool verbose = false)
      {
        using arr = Eigen::Array<T, 1, Eigen::Dynamic>;
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

  //! Different variants for defining the sub-problem in MALS-like algorithms for solving linear systems in TT format
  //!
  //! The sub-problem in each step is constructed using an orthogonal projection of the form W^T A V x = W^T b .
  //! This defines different choices for W.
  //!
  enum class MALS_projection
  {
    //! standard choice
    //!
    //! uses W = V, minimizes an energy functional for symmetric positive definite operators.
    //! Might still work for slightly non-symmetric operators.
    //!
    RitzGalerkin = 0,

    //! normal equations
    //!
    //! uses W = A V resulting in the Ritz-Galerkin approach for the normal equations: V^T A^TA V x = V^T A^T b.
    //! Suitable for non-symmetric operators but squares the condition number and doubles the TT ranks in the calculation.
    //! Can be interpreted as a Petrov-Galerkin approach with W = A V.
    //!
    NormalEquations,

    //! non-symmetric approach
    //!
    //! uses an orthogonal W that approximates AV, so A V = W B + E with W^T W = I and W^T E = 0.
    //! Slightly more work but avoids squaring the condition number and the TT ranks in the calculation.
    //! Might suffer from break-downs if the sub-problem operator B = W^T A V is not invertible.
    //!
    PetrovGalerkin
  };


  //! Solve a linear system using the MALS algorithm
  //!
  //! Approximate x with Ax = b
  //!
  //! @tparam T             data type (double, float, complex)
  //!
  //! @param TTOpA              tensor-train operator A
  //! @param symmetric          is the operator symmetric?
  //! @param projection         defines different variants for defining the local sub-problem in the MALS algorithm, for symmetric problems choose RitzGalerkin
  //! @param TTb                right-hand side tensor-train b
  //! @param TTx                initial guess on input, overwritten with the (approximate) result on output
  //! @param nSweeps            desired number of MALS sweeps
  //! @param residualTolerance  desired approximation accuracy, used to abort the iteration and to reduce the TTranks in the iteration
  //! @param maxRank            maximal allowed TT-rank, enforced even if this violates the residualTolerance
  //! @param nMALS              number of sub-tensors to combine as one local problem (1 for ALS, 2 for MALS, nDim for global GMRES)
  //! @param nOverlap           overlap (number of sub-tensors) of two consecutive local problems in one sweep (0 for ALS 1 for MALS, must be < nMALS)
  //! @param useTTgmres         use TT-GMRES for the local problem instead of normal GMRES with dense vectors
  //! @param gmresMaxITer       max. number of iterations for the inner (TT-)GMRES iteration
  //! @param gmresRelTol        relative residual tolerance for the inner (TT-)GMRES iteration
  //! @return                   residual norm of the result (||Ax - b||)
  //!
  template<typename T>
  T solveMALS(const TensorTrainOperator<T>& TTOpA,
              bool symmetric,
              const MALS_projection projection,
              const TensorTrain<T>& TTb,
              TensorTrain<T>& TTx,
              int nSweeps,
              T residualTolerance = std::sqrt(std::numeric_limits<T>::epsilon()),
              int maxRank = std::numeric_limits<int>::max(),
              int nMALS = 2, int nOverlap = 1,
              bool useTTgmres = false, int gmresMaxIter = 25, T gmresRelTol = T(1.e-4))
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
    constexpr auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());
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
    // temporary from orthogonalization
    Tensor2<T> tmp_left_v, tmp_right_v;

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

      Tensor3<T> tmp_last_left_v, tmp_last_right_v;
      if( projection == MALS_projection::RitzGalerkin )
      {
        left_v = TTx;
        right_v = TTx;
      }
      else // projection == MALS_projection::PetrovGalerkin
      {
        Tensor2<T> t2;
        for(int i = left_v_subT.size(); i < left_Ax.size(); i++)
        {
          Tensor3<T> newSubT;
          if( i > 0 )
            internal::normalize_contract1(tmp_left_v, left_Ax[i], newSubT);
          else
            copy(left_Ax[i], newSubT);
          unfold_left(newSubT, t2);
          auto [U,Vt] = internal::normalize_svd(t2, true);
          fold_left(U, newSubT.n(), newSubT);
          std::swap(tmp_left_v, Vt);
          left_v_subT.emplace_back(std::move(newSubT));
        }
        while(left_v_subT.size() > left_Ax.size())
          left_v_subT.pop_back();
        for(int i = right_v_subT.size(); i < right_Ax.size(); i++)
        {
          Tensor3<T> newSubT;
          if( i > 0 )
            internal::normalize_contract2(right_Ax[i], tmp_right_v, newSubT);
          else
            copy(right_Ax[i], newSubT);
          unfold_right(newSubT, t2);
          auto [U,Vt] = internal::normalize_svd(t2, false);
          fold_right(Vt, newSubT.n(), newSubT);
          std::swap(tmp_right_v, U);
          right_v_subT.emplace_back(std::move(newSubT));
        }
        while(right_v_subT.size() > right_Ax.size())
          right_v_subT.pop_back();
        

        // we need to enforce the size of the last sub-tensor in left/right_v_subT to obtain a quadratic local problem
        if( left_v_subT.size() > 0 )
        {
          const int left_r = TTx.subTensor(swpIdx.leftDim()).r1();
          if( left_v_subT.back().r2() < left_r )
            throw std::runtime_error("TensorTrain solveMALS: enlarging subspace for Petrov-Galerkin case not implemented yet!");
          tmp_last_left_v.resize(left_v_subT.back().r1(), left_v_subT.back().n(), left_r);
          for(int i = 0; i < tmp_last_left_v.r1(); i++)
            for (int j = 0; j < tmp_last_left_v.n(); j++)
              for (int k = 0; k < tmp_last_left_v.r2(); k++)
                tmp_last_left_v(i,j,k) = left_v_subT.back()(i,j,k);
          std::swap(tmp_last_left_v, left_v_subT.back());
          // when we truncate here, this also makes the last left_vTAx and left_vTb invalid
          while( left_vTAx.size() >= left_v_subT.size() )
            left_vTAx.pop_back();
          while( left_vTb.size() >= left_v_subT.size() )
            left_vTb.pop_back();
        }

        if( right_v_subT.size() >  0 )
        {
          const int right_r = TTx.subTensor(swpIdx.rightDim()).r2();
          if( right_v_subT.back().r1() < right_r )
            throw std::runtime_error("TensorTrain solveMALS: enlarging subspace for Petrov-Galerkin case not implemented yet!");
          tmp_last_right_v.resize(right_r, right_v_subT.back().n(), right_v_subT.back().r2());
          for(int i = 0; i < tmp_last_right_v.r1(); i++)
            for (int j = 0; j < tmp_last_right_v.n(); j++)
              for (int k = 0; k < tmp_last_right_v.r2(); k++)
                tmp_last_right_v(i,j,k) = right_v_subT.back()(i,j,k);
          std::swap(tmp_last_right_v, right_v_subT.back());
          // when we truncate here, this also makes the last right_vTAx and right_vTb invalid
          while( right_vTAx.size() >= right_v_subT.size() )
            right_vTAx.pop_back();
          while( right_vTb.size() >= right_v_subT.size() )
            right_vTb.pop_back();
        }

        left_v = left_v_subT;
        right_v = right_v_subT;
      }

      update_left_vTw<T>(left_v, TTb, 0, swpIdx.leftDim() - 1, left_vTb);
      update_left_vTw<T>(left_v, left_Ax, 0, swpIdx.leftDim() - 1, left_vTAx);

      update_right_vTw<T>(right_v, TTb, swpIdx.rightDim() + 1, nDim - 1, right_vTb);
      update_right_vTw<T>(right_v, right_Ax, swpIdx.rightDim() + 1, nDim - 1, right_vTAx);


      // prepare operator and right-hand side
      TensorTrain<T> tt_x = calculate_local_x(swpIdx.leftDim(), nMALS, TTx);
      const TensorTrain<T> tt_b = calculate_local_rhs(swpIdx.leftDim(), nMALS, left_vTb.back(), TTb, right_vTb.back());
      const TensorTrainOperator<T> localTTOp = calculate_local_op(swpIdx.leftDim(), nMALS, left_vTAx.back(), TTOpA, right_vTAx.back());
      if( projection == MALS_projection::RitzGalerkin )
      {
        assert(std::abs(dot(tt_x, tt_b) - dot(TTx, TTb)) <= sqrt_eps*std::abs(dot(TTx, TTb)));
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
        if( left_v_subT.size() > 0 )
        {
          std::swap(tmp_last_left_v, left_v_subT.back());
          // this also invalidates left_vTb and left_vTAx!
          left_vTb.pop_back();
          left_vTAx.pop_back();
        }
        if( right_v_subT.size() > 0 )
        {
          std::swap(tmp_last_right_v, right_v_subT.back());
          // this also invalidates right_vTb and right_vTAx!
          right_vTb.pop_back();
          right_vTAx.pop_back();
        }
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

        solveLocalProblem(swpIdx, iSweep == 0);
        lastSwpIdx = swpIdx;
      }
      // update remaining sub-tensors of left_Ax
      update_left_Ax(TTOpA, TTx, 0, nDim - 1, left_Ax);
      left_Ax = TTAx.setSubTensors(0, std::move(left_Ax));
      // for non-symm. cases, we still need left_Ax
      if( projection == MALS_projection::PetrovGalerkin )
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

        solveLocalProblem(swpIdx);
        lastSwpIdx = swpIdx;
      }
      // update remaining sub-tensors of right_Ax
      update_right_Ax(TTOpA, TTx, 0, nDim - 1, right_Ax);
      // TODO: that's wrong -> need a reverse
      right_Ax = TTAx.setSubTensors(0, reverse(std::move(right_Ax)));
      // for non-symm. cases, we still need right_Ax
      if( projection == MALS_projection::PetrovGalerkin )
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


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_HPP
