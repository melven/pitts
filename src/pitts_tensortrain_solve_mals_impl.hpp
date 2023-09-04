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
              int nMALS, int nOverlap, int nAMEnEnrichment, bool simplifiedAMEn,
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

      return solveMALS(TTOpAtA, true, MALS_projection::RitzGalerkin, TTAtb, TTx, nSweeps, residualTolerance, maxRank, nMALS, nOverlap, nAMEnEnrichment, simplifiedAMEn, useTTgmres, gmresMaxIter, gmresRelTol);
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


    // Generate index for sweeping (helper class)
    const int nDim = TTx.dimensions().size();
    nMALS = std::min(nMALS, nDim);
    nOverlap = std::min(nOverlap, nMALS-1);
    internal::SweepIndex lastSwpIdx(nDim, nMALS, nOverlap, -1);

    // trying to avoid costly residual calculations if possible, currently only for simplified AMEn
    const bool onlyLocalResidual = simplifiedAMEn && nMALS == 1 && projection == MALS_projection::RitzGalerkin;
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
      const T residualNorm_ref = norm2(TTOpA * TTx - TTb);
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

      TTx.setSubTensors(swpIdx.leftDim(), std::move(tt_x));
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

      if( !simplifiedAMEn )
      {
        // ensure indices are set correctly
        internal::ensureLeftOrtho_range(TTx, 0, swpIdx.leftDim());
        internal::ensureRightOrtho_range(TTx, swpIdx.rightDim(), nDim - 1);
        Ax.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
        Ax_ortho.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
        Ax_b_ortho.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
        vTAx.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
        vTb.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);

        const int iDim = swpIdx.leftDim();
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
      
      assert(simplifiedAMEn || check_AMEnSubspace(TTOpA, TTv, TTx, TTb, swpIdx, leftToRight, tt_z));

      if( leftToRight )
      {

        internal::ensureLeftOrtho_range(TTx, 0, swpIdx.rightDim()+1);

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

        const auto [Q, B] = internal::normalize_qb(unfold_left(subTr), true, T(0), nAMEnEnrichment);
        std::cout << " Enhancing subspace (left-to-right) for sub-tensor " << swpIdx.rightDim() << " for optimizing sub-tensor " << swpIdx.rightDim()+1 << ": increasing rank from " << subT0.r2() << " to " << subT0.r2()+Q.r2() << "\n";

        std::vector<Tensor3<T>> newSubT(2);
        newSubT[0].resize(subT0.r1(), subT0.n(), subT0.r2()+Q.r2());
        concatLeftRight<T>(unfold_left(subT0), Q, unfold_left(newSubT[0]));
        newSubT[1].resize(subT1.r1()+Q.r2(), subT1.n(), subT1.r2());
        concatTopBottom<T>(unfold_right(subT1), std::nullopt, unfold_right(newSubT[1]));
        //std::cout << "ortho:\n" << ConstEigenMap(unfold_left(newSubT[0])).transpose() * ConstEigenMap(unfold_left(newSubT[0])) << std::endl;
        TTx.setSubTensors(swpIdx.rightDim(), std::move(newSubT), {TT_Orthogonality::left, TT_Orthogonality::none});

        // these are not valid any more
        Ax.invalidate(swpIdx.rightDim()+1);
        Ax_ortho.invalidate(swpIdx.rightDim()+1);
        Ax_b_ortho.invalidate(swpIdx.rightDim()+1);
        vTAx.invalidate(swpIdx.rightDim()+1);
        vTb.invalidate(swpIdx.rightDim()+1);
      }
      else // right-to-left
      {
        internal::ensureRightOrtho_range(TTx, swpIdx.leftDim()-1, nDim-1);

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

        const auto [B, Qt] = internal::normalize_qb(unfold_right(subTr), false, T(0), nAMEnEnrichment);
        //std::cout << " Enhancing subspace (right-to-left) for sub-tensor " << swpIdx.leftDim() << " for optimizing sub-tensor " << swpIdx.leftDim()-1 << ": increasing rank from " << subT1.r1() << " to " << subT1.r1()+Qt.r1() << "\n";

        std::vector<Tensor3<T>> newSubT(2);
        newSubT[0].resize(subT0.r1(), subT0.n(), subT0.r2()+Qt.r1());
        concatLeftRight<T>(unfold_left(subT0), std::nullopt, unfold_left(newSubT[0]));
        newSubT[1].resize(subT1.r1()+Qt.r1(), subT1.n(), subT1.r2());
        concatTopBottom<T>(unfold_right(subT1), Qt, unfold_right(newSubT[1]));
        //std::cout << "ortho:\n" << ConstEigenMap(unfold_right(newSubT[1])) * ConstEigenMap(unfold_right(newSubT[1])).transpose() << std::endl;
        //std::cout << "added part:\n" << ConstEigenMap(Qt) << std::endl;
        TTx.setSubTensors(swpIdx.leftDim()-1, std::move(newSubT), {TT_Orthogonality::none, TT_Orthogonality::right});

        // these are not valid any more
        Ax.invalidate(swpIdx.leftDim()-1);
        Ax_ortho.invalidate(swpIdx.leftDim()-1);
        Ax_b_ortho.invalidate(swpIdx.leftDim()-1);
        vTAx.invalidate(swpIdx.leftDim()-1);
        vTb.invalidate(swpIdx.leftDim()-1);
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

        if( nAMEnEnrichment > 0 )
          enrichSubspace(swpIdx, true);
        
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
        const T residualNorm_ref = norm2(TTOpA * TTx - TTb);
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

        if( nAMEnEnrichment > 0 )
          enrichSubspace(swpIdx, false);

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
        const T residualNorm_ref = norm2(TTOpA * TTx - TTb);
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
