// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_solve_mals_impl.hpp
* @brief MALS algorithm for solving (non-)symmetric linear systems in TT format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-04-28
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SOLVE_MALS_IMPL_HPP
#define PITTS_TENSORTRAIN_SOLVE_MALS_IMPL_HPP

// includes
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>
#include "pitts_tensortrain_solve_mals.hpp"
#include "pitts_tensortrain_solve_mals_helper.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_concat.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_operator_apply_dense.hpp"
#include "pitts_tensortrain_operator_apply_transposed.hpp"
#include "pitts_tensortrain_operator_apply_op.hpp"
#include "pitts_tensortrain_operator_apply_transposed_op.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_axpby_normalized.hpp"
#include "pitts_tensortrain_solve_gmres.hpp"
#include "pitts_timer.hpp"
#include "pitts_tensortrain_sweep_index.hpp"
#ifndef NDEBUG
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_debug.hpp"
#include "pitts_tensortrain_operator_debug.hpp"
#include "pitts_tensortrain_solve_mals_debug.hpp"
#endif

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
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
              int nMALS, int nOverlap, int nAMEnEnrichment, bool simplifiedAMEn, int AMEn_ALS_residualRank,
              bool useTTgmres, int gmresMaxIter, T gmresRelTol, T estimatedConditionTTgmres)
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

      // ensure that we obtain a residual tolerance relative to TTb not TTAtb
      residualTolerance *= norm2(TTb) / norm2(TTAtb);

      return solveMALS(TTOpAtA, true, MALS_projection::RitzGalerkin, TTAtb, TTx, nSweeps, residualTolerance, maxRank, nMALS, nOverlap, nAMEnEnrichment, simplifiedAMEn, AMEn_ALS_residualRank, useTTgmres, gmresMaxIter, gmresRelTol);
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
    
    if( nAMEnEnrichment > 0 && nMALS > 1 )
      throw std::invalid_argument("TensorTrain solveMALS: currently AMEn is only supported for nMALS=1!");
    
    if( simplifiedAMEn && AMEn_ALS_residualRank > 0 )
      throw std::invalid_argument("Argument AMEn_ALS_residualRank only needed for simplifiedAMEn=false!");


    // Generate index for sweeping (helper class)
    const int nDim = TTx.dimensions().size();
    nMALS = std::min(nMALS, nDim);
    nOverlap = std::min(nOverlap, nMALS-1);
    internal::SweepIndex lastSwpIdx(nDim, nMALS, nOverlap, -1);

    // trying to avoid costly residual calculations if possible, currently only for simplified AMEn
    const bool AMEn_ALS = AMEn_ALS_residualRank > 0 && (!simplifiedAMEn);
    const bool onlyLocalResidual = (simplifiedAMEn || AMEn_ALS) && nMALS == 1 && projection == MALS_projection::RitzGalerkin;
    const T localResidualTolerance = residualTolerance / (2*std::sqrt(T(nDim-1)));

    const T nrm_TTb = onlyLocalResidual ? T(-1) : norm2(TTb);

#ifndef NDEBUG
    const auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());
#endif

    // check that 'symmetric' flag is used correctly
    assert(!symmetric || norm2((TTOpA-transpose(TTOpA)).tensorTrain()) < sqrt_eps);

    // for the Petrov-Galerkin variant, we get a dedicated left projection
    TensorTrain<T> TTw(0,0);
    const TensorTrain<T>& TTv = projection == MALS_projection::PetrovGalerkin ? TTw : TTx;

    // we store previous parts of w^Tb from left and right
    SweepData vTb = defineSweepData<Tensor2<T>>(nDim, dot_loop_from_left<T>(TTv, TTb), dot_loop_from_right<T>(TTv, TTb));
    
    // this includes a calculation of Ax, so allow to store the new parts of Ax in a seperate vector
    std::vector<Tensor3<T>> tmpAx(nDim);
    TensorTrain<T> TTAx(TTOpA.row_dimensions());
    SweepData Ax = defineSweepData<Tensor3<T>>(nDim, apply_loop(TTOpA, TTx), apply_loop(TTOpA, TTx));

    // left-/right orthogonal version of Ax
    SweepData Ax_ortho = defineSweepData<std::pair<Tensor3<T>,Tensor2<T>>>(nDim, ortho_loop_from_left<T>(Ax), ortho_loop_from_right<T>(Ax));

    // parts of Ax-b (left-/right orthogonalized)
    SweepData Ax_b_ortho = defineSweepData<std::pair<Tensor3<T>,Tensor2<T>>>(nDim, axpby_loop_from_left<T>(Ax_ortho, TTb), axpby_loop_from_right<T>(Ax_ortho, TTb));

    // we store previous parts of x^T A x
    SweepData vTAx = defineSweepData<Tensor2<T>>(nDim, dot_loop_from_left<T>(TTv, Ax), dot_loop_from_right<T>(TTv, Ax));

    // for AMEn+ALS, we update an approximation of the residual z \approx Ax - b
    TensorTrain<T> TTz(0,0);
    // set up AMEn+ALS initial residual
    if( AMEn_ALS )
    {
      TTz = TensorTrain<T>(TTOpA.row_dimensions(), AMEn_ALS_residualRank);
      randomize(TTz);
      internal::ensureRightOrtho_range(TTz, 0, nDim - 1);
    }

    // for AMEn+ALS, we store previous parts of z^T A x
    SweepData zTAx = defineSweepData<Tensor2<T>>(nDim, dot_loop_from_left<T>(TTz, Ax), dot_loop_from_right<T>(TTz, Ax));

    // for AMEn+ALS, we also need previous parts of z^T b
    SweepData zTb = defineSweepData<Tensor2<T>>(nDim, dot_loop_from_left<T>(TTz, TTb), dot_loop_from_right<T>(TTz, TTb));

    // for AMEn+ALS, we also need previous parts of V^T z
    SweepData vTz = defineSweepData<Tensor2<T>>(nDim, dot_loop_from_left<T>(TTv, TTz), dot_loop_from_right<T>(TTv, TTz));

    // store sub-tensor for enriching the subspace (AMEn)
    TensorTrain<T> tt_z(0,0);

    // calculate the error norm
    internal::ensureRightOrtho_range(TTx, 0, nDim - 1);
    Ax.update(-1, 0);
    assert(check_Ax(TTOpA, TTx, internal::SweepIndex(nDim, 1, 0, -1), Ax.data()));

    if( !onlyLocalResidual )
    {
      Ax_ortho.update(-1, 0);
      Ax_b_ortho.update(-1, 0);
      assert(check_Ax_ortho(TTOpA, TTx, Ax_ortho.data()));
      assert(check_Ax_b_ortho(TTOpA, TTx, TTb, Ax_ortho.data()[0].second(0,0), T(1), false, Ax_b_ortho.data()));
    }

    // calculate error
    T max_localResidualNorm = T(0);
    T residualNorm = nrm_TTb;
    if( !onlyLocalResidual )
    {
      const T norm_Ax = Ax_ortho.data().front().second(0,0);
      auto mapB = ConstEigenMap(Ax_b_ortho.data().front().second);
#ifndef PITTS_TENSORTRAIN_PLAIN_AXPBY
      residualNorm = (norm_Ax*Eigen::MatrixX<T>::Identity(mapB.rows(),mapB.cols()) - mapB).norm();
#else
      residualNorm = (norm_Ax*mapB.topRows(1) - mapB.bottomRows(1)).norm();
#endif
      assert(residualNorm - norm2(TTOpA * TTx - TTb) < sqrt_eps*nrm_TTb);
      std::cout << "Initial residual norm: " << residualNorm << " (abs), " << residualNorm / nrm_TTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";

      if( residualNorm / nrm_TTb < residualTolerance )
        return residualNorm;

    }

    // lambda to avoid code duplication: performs one step in a sweep
    const auto solveLocalProblem = [&](const internal::SweepIndex &swpIdx, bool firstSweep = false)
    {
      std::cout << " (M)ALS setup local problem for sub-tensors " << swpIdx.leftDim() << " to " << swpIdx.rightDim() << "\n";

      internal::ensureLeftOrtho_range(TTx, 0, swpIdx.leftDim());
      internal::ensureRightOrtho_range(TTx, swpIdx.rightDim(), nDim - 1);
      Ax.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
      if( !onlyLocalResidual )
      {
        Ax_ortho.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
        Ax_b_ortho.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
      }

      assert(check_Orthogonality(swpIdx, TTx));
      assert(check_Ax(TTOpA, TTx, swpIdx, Ax.data()));

      if( projection == MALS_projection::PetrovGalerkin )
      {
        // stupid implementation for testing
        TensorTrainOperator<T> TTv = setupProjectionOperator(TTx, swpIdx);
        TensorTrainOperator<T> TTAv(TTOpA.row_dimensions(), TTv.column_dimensions());
        apply(TTOpA, TTv, TTAv);

        assert(check_ProjectionOperator(TTOpA, TTx, swpIdx, TTv, TTAv));

        TTw = calculatePetrovGalerkinProjection(TTAv, swpIdx, TTx, true);

        assert(check_Orthogonality(swpIdx, TTw));

        // previously calculated projection differs (1. due to truncation and 2. because orthogonalize is not unique)
        vTAx.invalidate(0, nDim);
        vTb.invalidate(0, nDim);
      }

      vTAx.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
      vTb.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);

      // prepare operator and right-hand side
      TensorTrain<T> tt_x = calculate_local_x(swpIdx.leftDim(), nMALS, TTx);
      const TensorTrain<T> tt_b = calculate_local_rhs<T>(swpIdx.leftDim(), nMALS, vTb.left(), TTb, vTb.right());
      const TensorTrainOperator<T> localTTOp = calculate_local_op<T>(swpIdx.leftDim(), nMALS, vTAx.left(), TTOpA, vTAx.right());

      assert(check_systemDimensions(localTTOp, tt_x, tt_b));
      assert(check_localProblem(TTOpA, TTx, TTb, TTw, projection == MALS_projection::RitzGalerkin, swpIdx, localTTOp, tt_x, tt_b));

      T absTol, relTol;
      if( onlyLocalResidual )
      {
        // first Sweep: let GMRES start from zero
        if( firstSweep )
          tt_x.setZero();
        
        const T nrm_tt_b = norm2(tt_b);
        absTol = localResidualTolerance * nrm_tt_b;
        relTol = gmresRelTol;
      }
      else
      {
        // first Sweep: let GMRES start from zero
        if (firstSweep && residualNorm / nrm_TTb > 0.5)
          tt_x.setZero();
        
        absTol = gmresRelTol * residualTolerance * nrm_TTb;
        relTol = std::max(gmresRelTol, residualTolerance * nrm_TTb / residualNorm);
        // solveGMRES (TT-GMRES) uses the bigger one of the absolute and the relative tolerance (times norm(rhs)) for truncating the solution
        // This is too inaccurate close to the solution with the adaptive tolerance...
        if( useTTgmres )
          relTol = gmresRelTol; //std::min(T(0.1), relTol);
      }
      
      if( relTol < T(1) )
      {
        T localAbsRes, localRelRes;
        if (useTTgmres)
        {
          T estimatedCond = (nMALS == nDim) ? estimatedConditionTTgmres : 1;
          std::tie(localAbsRes, localRelRes) = solveGMRES(localTTOp, tt_b, tt_x, gmresMaxIter, absTol, relTol, estimatedCond, maxRank, true, symmetric, " (M)ALS local problem: ", true);
        }
        else
          std::tie(localAbsRes, localRelRes) = solveDenseGMRES(localTTOp, symmetric, tt_b, tt_x, maxRank, gmresMaxIter, absTol, relTol, " (M)ALS local problem: ", true);

        // max local residual *before* the iteration
        max_localResidualNorm = std::max(max_localResidualNorm, localAbsRes / localRelRes);
      }
      
      if( nAMEnEnrichment > 0 && simplifiedAMEn )
      {
        tt_z = TensorTrain<T>(tt_b.dimensions());
        apply(localTTOp, tt_x, tt_z);
        axpby(T(1), tt_b, T(-1), tt_z, T(0));
      }

      // update residual approximation with AMEn+ALS
      if( nAMEnEnrichment > 0 && AMEn_ALS )
      {
        internal::ensureLeftOrtho_range(TTz, 0, swpIdx.leftDim());
        internal::ensureRightOrtho_range(TTz, swpIdx.rightDim(), nDim - 1);
        zTAx.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
        zTb.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);

        const TensorTrain<T> z_tt_b = calculate_local_rhs<T>(swpIdx.leftDim(), nMALS, zTb.left(), TTb, zTb.right());
        const TensorTrainOperator<T> z_localTTOp = calculate_local_op<T>(swpIdx.leftDim(), nMALS, zTAx.left(), TTOpA, zTAx.right());

        assert(check_systemDimensions(z_localTTOp, tt_x, z_tt_b));
        assert(check_localProblem(TTOpA, TTx, TTb, TTz, false, swpIdx, z_localTTOp, tt_x, z_tt_b));

        tt_z = TensorTrain<T>(z_tt_b.dimensions());
        apply(z_localTTOp, tt_x, tt_z);
        tt_z.editSubTensor(0, [&](Tensor3<T>& t3_z){internal::t3_axpy<T>(T(-1), z_tt_b.subTensor(0), t3_z);});
      }

      TTx.setSubTensors(swpIdx.leftDim(), std::move(tt_x));

/*
      // invalidate everything that might depend on TTx (not really needed because orthogonal in the wrong direction in the next step -- but more clear)
      Ax.invalidate(swpIdx.leftDim(), nMALS);
      vTAx.invalidate(swpIdx.leftDim(), nMALS);
      vTb.invalidate(swpIdx.leftDim(), nMALS);
      if( !onlyLocalResidual )
      {
        Ax_ortho.invalidate(swpIdx.leftDim(), nMALS);
        Ax_b_ortho.invalidate(swpIdx.leftDim(), nMALS);
      }
*/
    };

    // AMEn idea: enrich subspace with some directions of the global residual (AMEn method)
    const auto enrichSubspace = [&](const internal::SweepIndex &swpIdx, bool leftToRight)
    {
      const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();
      if( leftToRight && swpIdx.rightDim() == swpIdx.nDim()-1 )
        return;
      if( (!leftToRight) && swpIdx.leftDim() == 0 )
        return;
      
      // only intended for nMALS == 1
      assert(swpIdx.leftDim() == swpIdx.rightDim());
      const int iDim = swpIdx.leftDim();

      if( (!simplifiedAMEn) && (!AMEn_ALS) )
      {
        // ensure indices are set correctly
        internal::ensureLeftOrtho_range(TTx, 0, swpIdx.leftDim());
        internal::ensureRightOrtho_range(TTx, swpIdx.rightDim(), nDim - 1);
        Ax.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
        Ax_ortho.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
        Ax_b_ortho.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
        vTAx.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
        vTb.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);

        Tensor3<T> tmpAx, tmpb, t3;
        internal::apply_contract(TTOpA, iDim, TTOpA.tensorTrain().subTensor(iDim), TTx.subTensor(iDim), tmpAx);
        std::vector<Tensor3<T>> subT(1);

        if( leftToRight )
        {
          if( iDim == 0 )
            copy(TTb.subTensor(iDim), tmpb);
          else
          {
            // first contraction: vTAx(*,:) * Ax(*,:,:)
            internal::reverse_dot_contract1<T>(*vTAx.left(), tmpAx, t3);
            std::swap(t3, tmpAx);
            // now contract: vTb(*,:) * b(*,:,:)
            internal::reverse_dot_contract1<T>(*vTb.left(), TTb.subTensor(iDim), tmpb);
          }
          internal::t3_scale(T(-1), tmpb);
          // contract: tmpAx(:,:,*) * B(*,:)
          const Tensor2<T>& prev_C = Ax_ortho.right()->get().second;
          internal::normalize_contract2(tmpAx, prev_C, t3);
          std::swap(t3, tmpAx);
          const Tensor2<T>& prev_B = Ax_b_ortho.right()->get().second;
#ifndef PITTS_TENSORTRAIN_PLAIN_AXPBY
          //         prev_B = (tmpAx tmpb)  * (I   0)
          //                                  (yTx R)
          // but top rows (I 0) are not stored...
          // => (tmpAx+tmpb*yTx   tmpb*R)
          internal::normalize_contract2(tmpb, prev_B, t3);
          EigenMap(unfold_left(t3)).leftCols(tmpAx.r2()) += ConstEigenMap(unfold_left(tmpAx));
          subT[0] = std::move(t3);
#else
          t3.resize(tmpAx.r1(), tmpAx.n(), tmpAx.r2()+tmpb.r2());
          concatLeftRight<T>(unfold_left(tmpAx), unfold_left(tmpb), unfold_left(t3));
          internal::normalize_contract2(t3, prev_B, subT[0]);
#endif
        }
        else // !leftToRight
        {
          if( iDim == nDim-1 )
            copy(TTb.subTensor(iDim), tmpb);
          else
          {
            // first contraction: Ax(:,:,*) * vTAx(:,*)
            internal::dot_contract1<T>(tmpAx, *vTAx.right(), t3);
            std::swap(t3, tmpAx);
            // now contract: b(:,:,*) * vTb(:,*)
            internal::dot_contract1<T>(TTb.subTensor(iDim), *vTb.right(), tmpb);
          }
          internal::t3_scale(T(-1), tmpb);
          // contract: B(:,*) * tmpAx(*,:,:)
          const Tensor2<T>& prev_C = Ax_ortho.left()->get().second;
          internal::normalize_contract1(prev_C, tmpAx, t3);
          std::swap(t3, tmpAx);
          const Tensor2<T>& prev_B = Ax_b_ortho.left()->get().second;
#ifndef PITTS_TENSORTRAIN_PLAIN_AXPBY
          //         prev_B = (I xTy)    *    (tmpAx)
          //                  (0  R )         (tmpb)
          // but left columns (I;0) are not stored...
          // => (tmpAx+xTy*tmpb)
          //    (R*tmpb)
          internal::normalize_contract1(prev_B, tmpb, t3);
          EigenMap(unfold_right(t3)).topRows(tmpAx.r1()) += ConstEigenMap(unfold_right(tmpAx));
          subT[0] = std::move(t3);
#else
          t3.resize(tmpAx.r1()+tmpb.r1(), tmpAx.n(), tmpAx.r2());
          concatTopBottom<T>(unfold_right(tmpAx), unfold_right(tmpb), unfold_right(t3));
          internal::normalize_contract1(prev_B, t3, subT[0]);
#endif
        }
        tt_z = TensorTrain<T>(std::move(subT));
      }

      if( AMEn_ALS )
      {
        TTz.setSubTensors(swpIdx.leftDim(), std::move(tt_z));
        zTAx.invalidate(swpIdx.leftDim(), nMALS);
        zTb.invalidate(swpIdx.leftDim(), nMALS);
        vTz.invalidate(swpIdx.leftDim(), nMALS);

        vTz.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);

        std::vector<Tensor3<T>> subT(1);

        if( leftToRight )
        {
          if( iDim == 0 )
            copy(TTz.subTensor(iDim), subT[0]);
          else
          {
            // contraction: vTz(*,:) * TTz(*,:,:)
            internal::reverse_dot_contract1<T>(*vTz.left(), TTz.subTensor(iDim), subT[0]);
          }
        }
        else // !leftToRight
        {
          if( iDim == nDim-1 )
            copy(TTz.subTensor(iDim), subT[0]);
          else
          {
            // contraction: TTz(:,:,*) * vTz(:,*)
            internal::dot_contract1<T>(TTz.subTensor(iDim), *vTz.right(), subT[0]);
          }
        }
        tt_z = TensorTrain<T>(std::move(subT));
      }
      
      if( !simplifiedAMEn )
      {
        if( AMEn_ALS )
          assert(check_AMEn_ALS_Subspace(TTOpA, TTv, TTx, TTb, swpIdx, leftToRight, tt_z, TTz));
        else
          assert(check_AMEnSubspace(TTOpA, TTv, TTx, TTb, swpIdx, leftToRight, tt_z));
      }

      if( leftToRight )
      {

        internal::ensureLeftOrtho_range(TTx, 0, swpIdx.rightDim()+1);

        // these are not valid any more
        Ax.invalidate(swpIdx.rightDim()+1);
        Ax_ortho.invalidate(swpIdx.rightDim()+1);
        Ax_b_ortho.invalidate(swpIdx.rightDim()+1);
        vTAx.invalidate(swpIdx.rightDim()+1);
        vTb.invalidate(swpIdx.rightDim()+1);
        if( AMEn_ALS )
          vTz.invalidate(swpIdx.rightDim()+1);

        const Tensor3<T>& subT0 = TTx.subTensor(swpIdx.rightDim());
        const Tensor3<T>& subT1 = TTx.subTensor(swpIdx.rightDim()+1);
        Tensor3<T> subTr = tt_z.setSubTensor(0, Tensor3<T>(1,tt_z.dimensions()[0],1));

        // orthogonalize wrt. subT0
        const T zNorm = internal::t3_nrm(subTr);
        if( zNorm > T(0) )
          internal::t3_scale(T(1)/zNorm, subTr);
        Tensor2<T> tmp;
        internal::reverse_dot_contract2(subT0, subTr, tmp); // subT0(*,*,:) * subTr(*,*,:)
        dot_contract1t_sub(subT0, tmp, subTr); // subTr(:,:,:) -= subT0(:,:,*) * tmp(*,:)
        const T zNorm2 = internal::t3_nrm(subTr);
        //std::cout << "zNorm before/after: " << zNorm << " " << zNorm2 << std::endl;
        if( zNorm2 <= std::numeric_limits<T>::epsilon()*std::sqrt(subT1.r1()*subT1.n())*1000 )
        {
          std::cout << " Not enhancing subspace (left-to-right) for sub-tensor " << swpIdx.rightDim() << " as the residual does not contain any new directions!\n";
          return;
        }

        int maxEnrichmentRank = nAMEnEnrichment;
        maxEnrichmentRank = std::min<int>(maxEnrichmentRank, subT0.r1()*subT0.n()-subT0.r2());
        maxEnrichmentRank = std::min<int>(maxEnrichmentRank, subT1.r2()*subT1.n()-subT1.r1());
        const auto [Q, B] = [&]()
        {
          if( maxEnrichmentRank < subTr.r2() && maxEnrichmentRank < subTr.r1()*subTr.n() )
            return internal::normalize_svd(unfold_left(subTr), true, T(0), maxEnrichmentRank, true);
          
          return internal::normalize_qb(unfold_left(subTr), true, T(0), maxEnrichmentRank, true);
        }();
        std::cout << " Enhancing subspace (left-to-right) for sub-tensor " << swpIdx.rightDim() << " for optimizing sub-tensor " << swpIdx.rightDim()+1 << ": increasing rank from " << subT0.r2() << " to " << subT0.r2()+Q.r2() << "\n";

        std::vector<Tensor3<T>> newSubT(2);
        newSubT[0].resize(subT0.r1(), subT0.n(), subT0.r2()+Q.r2());
        concatLeftRight<T>(unfold_left(subT0), Q, unfold_left(newSubT[0]));
        newSubT[1].resize(subT1.r1()+Q.r2(), subT1.n(), subT1.r2());
        concatTopBottom<T>(unfold_right(subT1), std::nullopt, unfold_right(newSubT[1]));
        //std::cout << "ortho:\n" << ConstEigenMap(unfold_left(newSubT[0])).transpose() * ConstEigenMap(unfold_left(newSubT[0])) << std::endl;
        TTx.setSubTensors(swpIdx.rightDim(), std::move(newSubT), {TT_Orthogonality::left, TT_Orthogonality::none});
      }
      else // right-to-left
      {
        internal::ensureRightOrtho_range(TTx, swpIdx.leftDim()-1, nDim-1);

        // these are not valid any more
        Ax.invalidate(swpIdx.leftDim()-1);
        Ax_ortho.invalidate(swpIdx.leftDim()-1);
        Ax_b_ortho.invalidate(swpIdx.leftDim()-1);
        vTAx.invalidate(swpIdx.leftDim()-1);
        vTb.invalidate(swpIdx.leftDim()-1);
        if( AMEn_ALS )
          vTz.invalidate(swpIdx.leftDim()-1);

        const Tensor3<T>& subT0 = TTx.subTensor(swpIdx.leftDim()-1);
        const Tensor3<T>& subT1 = TTx.subTensor(swpIdx.leftDim());
        Tensor3<T> subTr = tt_z.setSubTensor(0, Tensor3<T>(1,tt_z.dimensions()[0],1));

        // orthogonalize wrt. subT1
        const T zNorm = internal::t3_nrm(subTr);
        if( zNorm > T(0) )
          internal::t3_scale(T(1)/zNorm, subTr);
        Tensor2<T> tmp;
        internal::dot_contract2(subT1, subTr, tmp); // subT1(:,*,*) * subTr(:,*,*)
        reverse_dot_contract1_sub(tmp, subT1, subTr); // subTr(:,:,:) -= tmp(*,:) * subT1(*,:,:)
        const T zNorm2 = internal::t3_nrm(subTr);
        //std::cout << "zNorm before/after: " << zNorm << " " << zNorm2 << std::endl;
        if( zNorm2 <= std::numeric_limits<T>::epsilon()*std::sqrt(subT1.r1()*subT1.n())*1000 )
        {
          std::cout << " Not enhancing subspace (right-to-left) for sub-tensor " << swpIdx.leftDim() << " as the residual does not contain any new directions!\n";
          return;
        }

        int maxEnrichmentRank = nAMEnEnrichment;
        maxEnrichmentRank = std::min<int>(maxEnrichmentRank, subT0.r1()*subT0.n()-subT0.r2());
        maxEnrichmentRank = std::min<int>(maxEnrichmentRank, subT1.r2()*subT1.n()-subT1.r1());
        const auto [B, Qt] = [&]()
        {
          if( maxEnrichmentRank < subTr.r1() && maxEnrichmentRank < subTr.n()*subTr.r2() )
            return internal::normalize_svd(unfold_right(subTr), false, T(0), maxEnrichmentRank, true);
          
          return internal::normalize_qb(unfold_right(subTr), false, T(0), maxEnrichmentRank, true);
        }();
        std::cout << " Enhancing subspace (right-to-left) for sub-tensor " << swpIdx.leftDim() << " for optimizing sub-tensor " << swpIdx.leftDim()-1 << ": increasing rank from " << subT1.r1() << " to " << subT1.r1()+Qt.r1() << "\n";

        std::vector<Tensor3<T>> newSubT(2);
        newSubT[0].resize(subT0.r1(), subT0.n(), subT0.r2()+Qt.r1());
        concatLeftRight<T>(unfold_left(subT0), std::nullopt, unfold_left(newSubT[0]));
        newSubT[1].resize(subT1.r1()+Qt.r1(), subT1.n(), subT1.r2());
        concatTopBottom<T>(unfold_right(subT1), Qt, unfold_right(newSubT[1]));
        //std::cout << "ortho:\n" << ConstEigenMap(unfold_right(newSubT[1])) * ConstEigenMap(unfold_right(newSubT[1])).transpose() << std::endl;
        //std::cout << "added part:\n" << ConstEigenMap(Qt) << std::endl;
        TTx.setSubTensors(swpIdx.leftDim()-1, std::move(newSubT), {TT_Orthogonality::none, TT_Orthogonality::right});
      }
    };

    // now everything is prepared, perform the sweeps
    for(int iSweep = 0; iSweep < nSweeps; iSweep++)
    {
      // sweep left to right
      max_localResidualNorm = T(0);
      for(auto swpIdx = lastSwpIdx.first(); swpIdx; swpIdx = swpIdx.next())
      {
        // skip iteration if this is the same as in the last right-to-left sweep
        if( nMALS == nDim || swpIdx != lastSwpIdx )
          solveLocalProblem(swpIdx, iSweep == 0);

#ifndef NDEBUG
  TensorTrain<T> TTx_beforeEnrichment(TTx.dimensions());
  copy(TTx, TTx_beforeEnrichment);
#endif

        if( nAMEnEnrichment > 0 )
          enrichSubspace(swpIdx, true);
        
#ifndef NDEBUG
  const T xDiff = axpby(T(-1), TTx, T(1), TTx_beforeEnrichment, T(0));
  const T xNorm = norm2(TTx_beforeEnrichment);
  std::cout << "Enrichment error: " << xDiff << ", xNorm: " << xNorm << "\n";
  //assert(std::abs(xDiff) <= 10*std::abs(xNorm)*sqrt_eps);
#endif

        lastSwpIdx = swpIdx;
      }
      // update remaining sub-tensors of Ax
      Ax.update(nDim-1, nDim);
      assert(check_Ax(TTOpA, TTx, internal::SweepIndex(nDim, 1, 0, nDim), Ax.data()));

      if( onlyLocalResidual )
      {
        std::cout << "Sweep " << iSweep+0.5 << " max local residual norm: " << max_localResidualNorm << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";
        if( max_localResidualNorm < residualTolerance )
          break;
      }
      else // !onlyLocalResidual
      {
        Ax_ortho.update(nDim-1, nDim);
        Ax_b_ortho.update(nDim-1, nDim);
        assert(check_Ax_ortho(TTOpA, TTx, Ax_ortho.data()));
        assert(check_Ax_b_ortho(TTOpA, TTx, TTb, Ax_ortho.data()[nDim-1].second(0,0), T(1), true, Ax_b_ortho.data()));
      
        // check error
        const T norm_Ax = Ax_ortho.data().back().second(0,0);
        auto mapB = ConstEigenMap(Ax_b_ortho.data().back().second);
#ifndef PITTS_TENSORTRAIN_PLAIN_AXPBY
        residualNorm = (norm_Ax*Eigen::MatrixX<T>::Identity(mapB.rows(),mapB.cols()) - mapB).norm();
#else
        residualNorm = (norm_Ax*mapB.leftCols(1) - mapB.rightCols(1)).norm();
#endif
        assert(residualNorm - norm2(TTOpA * TTx - TTb) < sqrt_eps*nrm_TTb);
        std::cout << "Sweep " << iSweep+0.5 << " residual norm: " << residualNorm << " (abs), " << residualNorm / nrm_TTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";

        if( residualNorm / nrm_TTb < residualTolerance )
          break;
      }


      // sweep right to left
      max_localResidualNorm = T(0);
      for(auto swpIdx = lastSwpIdx.last(); swpIdx; swpIdx = swpIdx.previous())
      {
        // skip iteration if this is the same as in the last left-to-right sweep
        if( swpIdx != lastSwpIdx )
          solveLocalProblem(swpIdx);

#ifndef NDEBUG
  TensorTrain<T> TTx_beforeEnrichment(TTx.dimensions());
  copy(TTx, TTx_beforeEnrichment);
#endif

        if( nAMEnEnrichment > 0 )
          enrichSubspace(swpIdx, false);

#ifndef NDEBUG
  const T xDiff = axpby(T(-1), TTx, T(1), TTx_beforeEnrichment, T(0));
  const T xNorm = norm2(TTx_beforeEnrichment);
  std::cout << "Enrichment error: " << xDiff << ", xNorm: " << xNorm << "\n";
  //assert(std::abs(xDiff) <= std::abs(xNorm)*sqrt_eps);
#endif

        lastSwpIdx = swpIdx;
      }
      // update remaining sub-tensors of right_Ax
      Ax.update(-1, 0);
      assert(check_Ax(TTOpA, TTx, internal::SweepIndex(nDim, 1, 0, -1), Ax.data()));

      if( onlyLocalResidual )
      {
        std::cout << "Sweep " << iSweep+1 << " max local residual norm: " << max_localResidualNorm << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";
        if( max_localResidualNorm < residualTolerance )
          break;
      }
      else // !onlyLocalResidual
      {
        Ax_ortho.update(-1, 0);
        Ax_b_ortho.update(-1, 0);
        assert(check_Ax_ortho(TTOpA, TTx, Ax_ortho.data()));
        assert(check_Ax_b_ortho(TTOpA, TTx, TTb, Ax_ortho.data()[0].second(0,0), T(1), false, Ax_b_ortho.data()));

        // check error
        const T norm_Ax = Ax_ortho.data().front().second(0,0);
        auto mapB = ConstEigenMap(Ax_b_ortho.data().front().second);
#ifndef PITTS_TENSORTRAIN_PLAIN_AXPBY
        residualNorm = (norm_Ax*Eigen::MatrixX<T>::Identity(mapB.rows(),mapB.cols()) - mapB).norm();
#else
        residualNorm = (norm_Ax*mapB.topRows(1) - mapB.bottomRows(1)).norm();
#endif
        assert(residualNorm - norm2(TTOpA * TTx - TTb) < sqrt_eps*nrm_TTb);
        std::cout << "Sweep " << iSweep+1 << " residual norm: " << residualNorm << " (abs), " << residualNorm / nrm_TTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";

        if( residualNorm / nrm_TTb < residualTolerance )
          break;
      }
    }

/*
    std::cout << "Ax ranks:\n";
    for(const auto& subT: Ax.data())
      std::cout << "   " << subT.r1() << " x " << subT.r2();
    std::cout << "\n";

    std::cout << "Ax_ortho ranks:\n";
    for(const auto& [Q, B]: Ax_ortho.data())
      std::cout << "   " << Q.r1() << " x " << Q.r2();
    std::cout << "\n";
    std::cout << "Ax_ortho B diag:\n";
    for(const auto& [Q, B]: Ax_ortho.data())
      std::cout << ConstEigenMap(B).bdcSvd().singularValues().transpose() << "\n";


    std::cout << "Ax_b_ortho ranks:\n";
    for(const auto& [Q, B]: Ax_b_ortho.data())
      std::cout << "   " << Q.r1() << " x " << Q.r2();
    std::cout << "\n";
    std::cout << "Ax_b_ortho B diag:\n";
    for(const auto& [Q, B]: Ax_b_ortho.data())
      std::cout << ConstEigenMap(B).bdcSvd().singularValues().transpose() << "\n";
*/

    return onlyLocalResidual ? max_localResidualNorm : residualNorm;
  }

}


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_IMPL_HPP
