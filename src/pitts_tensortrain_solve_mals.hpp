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
      //! calculate next part of x^Tb from right to left or discard last part
      //!
      //! Like TT dot product bug allows to store all intermediate results.
      //!
      //! If TTOpA is provided, also calculate next part of b^TAx from right to left
      //! again like TT dot product but with Ax
      //! we have
      //!  |   |     |
      //!  -- bTAx --
      //!
      //! and we need for the next step
      //!   |        |      |
      //!  b_k^T -- A_k -- x_k
      //!   |        |      |
      //!   ------ bTAx -----
      //!
      //! Also calculates the current sub-tensor of A*x
      template<typename T>
      void update_right_xTb(const TensorTrain<T>& TTb, const TensorTrain<T>& TTx, int firstIdx, int lastIdx, std::vector<Tensor2<T>>& right_xTb,
                            const TensorTrainOperator<T>* TTOpA = nullptr, std::vector<Tensor3<T>>* TTAx = nullptr)
      {
        assert((TTOpA == nullptr) == (TTAx == nullptr));
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int nDim = TTb.dimensions().size();
        assert(nDim == TTx.dimensions().size());
        assert(0 <= firstIdx);
        assert(firstIdx <= lastIdx+1);
        assert(lastIdx == nDim-1);

        // first call? right_xTb should at least contain a 1x1 one tensor
        if( right_xTb.empty() )
        {
          right_xTb.emplace_back(Tensor2<T>{1,1});
          right_xTb.back()(0,0) = T(1);
        }
        if( TTOpA && TTAx->empty() )
          TTAx->resize(nDim);

        // calculate new entries in right_xTb when sweeping right-to-left
        for(int iDim = lastIdx - (right_xTb.size()-1); iDim >= firstIdx; iDim--)
        {
          if( TTOpA )
          {
            const auto& subTx = TTx.subTensor(iDim);
            const auto& subTOpA = TTOpA->tensorTrain().subTensor(iDim);
            // first contract A_k with x_k
            //     |      |
            // -- A_k -- x_k
            //     |      |
            //
            internal::apply_contract(*TTOpA, iDim, subTOpA, subTx, TTAx->at(iDim));
            // now we have
            //   |        ||
            //  b_k^T -- Axk
            //   |        ||
            //   ------ bTAx
          }

          const auto& subTb = TTb.subTensor(iDim);
          const auto& subTAx = TTOpA ? TTAx->at(iDim) : TTx.subTensor(iDim);

          // first contraction: subTb(:,:,*) * prev_t2(:,*)
          Tensor3<T> t3_tmp;
          internal::dot_contract1(subTb, right_xTb.back(), t3_tmp);

          // second contraction: subTAx(:,*,*) * t3_tmp(:,*,*)
          Tensor2<T> t2;
          internal::dot_contract2(subTAx, t3_tmp, t2);
          right_xTb.emplace_back(std::move(t2));
        }

        // discard old entries in right_xTb when sweeping left-to-right
        for(int iDim = lastIdx - (right_xTb.size()-1); iDim+1 < firstIdx; iDim++)
          right_xTb.pop_back();
      }

      //! calculate next part of x^Tb from left to right or discard last part
      //!
      //! Like TT dot product bug allows to store all intermediate results.
      //!
      //! If TTOpA is provided, also calculate next part of b^TAx from left to right
      //!
      //! Also calculates the current sub-tensor of A*x
      //!
      template<typename T>
      void update_left_xTb(const TensorTrain<T>& TTb, const TensorTrain<T>& TTx, int firstIdx, int lastIdx, std::vector<Tensor2<T>>& left_xTb,
                           const TensorTrainOperator<T>* TTOpA = nullptr, std::vector<Tensor3<T>>* TTAx = nullptr)
      {
        assert((TTOpA == nullptr) == (TTAx == nullptr));
        const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

        const int nDim = TTb.dimensions().size();
        assert(nDim == TTx.dimensions().size());
        assert(0 == firstIdx);
        assert(firstIdx-1 <= lastIdx);
        assert(lastIdx < nDim);

        // first call? left_xTb should at least contain a 1x1 one tensor
        if( left_xTb.empty() )
        {
          left_xTb.emplace_back(Tensor2<T>{1,1});
          left_xTb.back()(0,0) = T(1);
        }
        if( TTOpA && TTAx->empty() )
          TTAx->resize(nDim);

        // calculate new entries in left_xTb when sweeping left-to-right

        for(int iDim = firstIdx + (left_xTb.size()-1); iDim <= lastIdx; iDim++)
        {
          if( TTOpA )
          {
            const auto& subTOpA = TTOpA->tensorTrain().subTensor(iDim);
            const auto& subTx = TTx.subTensor(iDim);
            internal::apply_contract(*TTOpA, iDim, subTOpA, subTx, TTAx->at(iDim));
            // now we have
            //   ------ xTAx
            //   |        ||
            //  x_k^T -- Axk
            //   |        ||
          }

          const auto& subTb = TTb.subTensor(iDim);
          const auto& subTAx = TTOpA ? TTAx->at(iDim) : TTx.subTensor(iDim);

          // first contraction: pve_t2(*,:) * subTb(*,:,:)
          Tensor3<T> t3_tmp;
          internal::reverse_dot_contract1(left_xTb.back(), subTb, t3_tmp);

          // second contraction: t3(*,*,:) * subTAx(*,*,:)
          Tensor2<T> t2;
          internal::reverse_dot_contract2(t3_tmp, subTAx, t2);
          left_xTb.emplace_back(std::move(t2));
        }

        // discard old entries in left_xTb when sweeping right-to-left
        for(int iDim = firstIdx + (left_xTb.size()-1); iDim-1 > lastIdx; iDim--)
          left_xTb.pop_back();
      }

      //! calculate the local RHS tensor-train for (M)ALS
      template<typename T>
      TensorTrain<T> calculate_local_rhs(int iDim, int nMALS, const Tensor2<T>& left_xTb, const TensorTrain<T>& TTb, const Tensor2<T>& right_xTb)
      {
        std::vector<Tensor3<T>> subT_b(nMALS);
        for(int i = 0; i < nMALS; i++)
          copy(TTb.subTensor(iDim+i), subT_b[i]);

        // first contract: tt_b_right(:,:,*) * right_xTb(:,*)
        Tensor3<T> t3_tmp;
        std::swap(subT_b.back(), t3_tmp);
        internal::dot_contract1(t3_tmp, right_xTb, subT_b.back());

        // then contract: left_xTb(*,:) * tt_b_left(*,:,:)
        std::swap(subT_b.front(), t3_tmp);
        internal::reverse_dot_contract1(left_xTb, t3_tmp, subT_b.front());

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
          const int r1 = t2.r1();
          const int rAl = t2.r2() / r1;

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
                t3(0,i+j*r1,k) = t2(i,j+k*r1);
      }

      template<typename T>
      void copy_op_right(const Tensor2<T>& t2, Tensor3<T>& t3)
      {
          const int r2 = t2.r2();
          const int rAr = t2.r1() / r2;

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
                t3(k,i+j*r2,0) = t2(j+k*r2,i);
      }

      //! calculate the local linear operator in TT format for (M)ALS
      template<typename T>
      TensorTrainOperator<T> calculate_local_op(int iDim, int nMALS, const Tensor2<T>& left_xTAx, const TensorTrainOperator<T>& TTOp, const Tensor2<T>& right_xTAx)
      {
        const int n0 = left_xTAx.r1();
        const int nd = right_xTAx.r2();

        std::vector<int> localRowDims(nMALS+2), localColDims(nMALS+2);
        localRowDims.front() = localColDims.front() = n0;
        for(int i = 0; i < nMALS; i++)
        {
          localRowDims[i+1] = TTOp.row_dimensions()[iDim+i];
          localColDims[i+1] = TTOp.column_dimensions()[iDim+i];
        }
        localRowDims.back()  = localColDims.back()  = nd;

        std::vector<Tensor3<T>> subT_localOp(nMALS+2);
        copy_op_left(left_xTAx, subT_localOp.front());
        for(int i = 0; i < nMALS; i++)
          copy(TTOp.tensorTrain().subTensor(iDim+i), subT_localOp[1+i]);
        copy_op_right(right_xTAx, subT_localOp.back());

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
        const arr localRes = GMRES<arr>(tt_OpA, true, mv_rhs, mv_x, maxIter, arr::Constant(1, absTol), arr::Constant(1, relTol), outputPrefix, verbose);

        const auto r_left = tt_x.subTensor(0).r1();
        const auto r_right = tt_x.subTensor(nDim-1).r2();
        TensorTrain<T> new_tt_x = fromDense(mv_x, mv_rhs, tt_x.dimensions(), relTol/nDim, maxRank, false, r_left, r_right);
        std::swap(tt_x, new_tt_x);

        return localRes(0);
      }
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

    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // check that dimensions match
    if( TTb.dimensions() != TTOpA.row_dimensions() )
      throw std::invalid_argument("TensorTrain solveMALS: operator and rhs dimensions mismatch!");
    if( TTx.dimensions() != TTOpA.column_dimensions() )
      throw std::invalid_argument("TensorTrain solveMALS: operator and x dimensions mismatch!");

    // Generate index for sweeping (helper class)
    const int nDim = TTx.dimensions().size();
    nMALS = std::min(nMALS, nDim);
    nOverlap = std::min(nOverlap, nMALS-1);
    internal::SweepIndex lastSwpIdx(nDim, nMALS, nOverlap, -1);

    // for the non-symmetric case, we solve the normal equations, so calculate A^T*b and A^T*A
    // and provide convenient name for TTOpA resp. TTOpAtA for the code below
    const auto& effTTb = [&]()
    {
      if( projection == MALS_projection::NormalEquations )
      {
        TensorTrain<T> TTAtb(TTOpA.column_dimensions());
        applyT(TTOpA, TTb, TTAtb);
        return TTAtb;
      }
      else
      {
        return TTb;
      }
    }();
    const auto& effTTOpA = [&]()
    {
      if( projection == MALS_projection::NormalEquations )
      {
        TensorTrainOperator<T> TTOpAtA(TTOpA.column_dimensions(), TTOpA.column_dimensions());
        applyT(TTOpA, TTOpA, TTOpAtA);
        return TTOpAtA;
      }
      else
      {
        return TTOpA;
      }
    }();

    const T nrm_TTb = norm2(effTTb);

#ifndef NDEBUG
    constexpr auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());
#endif

    // we store previous parts of x^Tb from left and right
    // (respectively x^T A^T b for the non-symmetric case)
    std::vector<Tensor2<T>> left_xTb, right_xTb;
    
    // we store previous parts of x^T A x
    // (respectively x^T A^T A x for the non-symmetric case)
    std::vector<Tensor2<T>> left_xTAx, right_xTAx;

    // this includes a calculation of Ax, so allow to store the new parts of Ax in a seperate vector
    std::vector<Tensor3<T>> TTAx_subT;
    TensorTrain<T> TTAx(effTTOpA.row_dimensions());

    // calculate the error norm
    apply(effTTOpA, TTx, TTAx);
    T residualNorm = axpby(T(1), effTTb, T(-1), TTAx);
    std::cout << "Initial residual norm: " << residualNorm << " (abs), " << residualNorm / nrm_TTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";

    // lambda to avoid code duplication: performs one step in a sweep
    const auto solveLocalProblem = [&](const internal::SweepIndex &swpIdx, bool firstSweep = false)
    {
      internal::ensureLeftOrtho_range(TTx, 0, swpIdx.leftDim());
      update_left_xTb(effTTb, TTx, 0, swpIdx.leftDim() - 1, left_xTb);
      update_left_xTb(TTx, TTx, 0, swpIdx.leftDim() - 1, left_xTAx, &effTTOpA, &TTAx_subT);

      internal::ensureRightOrtho_range(TTx, swpIdx.rightDim(), nDim - 1);
      update_right_xTb(effTTb, TTx, swpIdx.rightDim() + 1, nDim - 1, right_xTb);
      update_right_xTb(TTx, TTx, swpIdx.rightDim() + 1, nDim - 1, right_xTAx, &effTTOpA, &TTAx_subT);

      std::cout << " (M)ALS setup local problem for sub-tensors " << swpIdx.leftDim() << " to " << swpIdx.rightDim() << "\n";

      // prepare operator and right-hand side
      TensorTrain<T> tt_x = calculate_local_x(swpIdx.leftDim(), nMALS, TTx);
      const TensorTrain<T> tt_b = calculate_local_rhs(swpIdx.leftDim(), nMALS, left_xTb.back(), effTTb, right_xTb.back());
      const TensorTrainOperator<T> localTTOp = calculate_local_op(swpIdx.leftDim(), nMALS, left_xTAx.back(), effTTOpA, right_xTAx.back());
      assert(std::abs(dot(tt_x, tt_b) - dot(TTx, effTTb)) < sqrt_eps);
      // first Sweep: let GMRES start from zero, at least favorable for TT-GMRES!
      if (firstSweep && residualNorm / nrm_TTb > 0.5)
        tt_x.setZero();

      if (useTTgmres)
        const T localRes = solveGMRES(localTTOp, tt_b, tt_x, gmresMaxIter, gmresRelTol * residualTolerance * nrm_TTb, gmresRelTol, maxRank, true, true, " (M)ALS local problem: ", true);
      else
        const T localRes = solveDenseGMRES(localTTOp, true, tt_b, tt_x, maxRank, gmresMaxIter, gmresRelTol * residualTolerance * nrm_TTb, gmresRelTol, " (M)ALS local problem: ", true);

      TTx.setSubTensors(swpIdx.leftDim(), std::move(tt_x));
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
      // update remaining sub-tensors of TTAx
      for(int iDim = lastSwpIdx.leftDim(); iDim < nDim; iDim++)
        internal::apply_contract(effTTOpA, iDim, effTTOpA.tensorTrain().subTensor(iDim), TTx.subTensor(iDim), TTAx_subT.at(iDim));
      TTAx_subT = TTAx.setSubTensors(0, std::move(TTAx_subT));

      assert( norm2(effTTOpA * TTx - TTAx) < sqrt_eps );

      // check error
      residualNorm = axpby(T(1), effTTb, T(-1), TTAx);
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
      for(int iDim = lastSwpIdx.rightDim(); iDim >= 0; iDim--)
        internal::apply_contract(effTTOpA, iDim, effTTOpA.tensorTrain().subTensor(iDim), TTx.subTensor(iDim), TTAx_subT.at(iDim));
      TTAx_subT = TTAx.setSubTensors(0, std::move(TTAx_subT));

      assert(norm2(effTTOpA * TTx - TTAx) < sqrt_eps);

      // check error
      residualNorm = axpby(T(1), effTTb, T(-1), TTAx);
      std::cout << "Sweep " << iSweep+1 << " residual norm: " << residualNorm << " (abs), " << residualNorm / nrm_TTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";
    }


    return residualNorm;
  }

}


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_HPP
