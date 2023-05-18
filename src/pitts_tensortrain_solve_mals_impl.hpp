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
              int nMALS, int nOverlap, int nAMEnEnrichment,
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

      return solveMALS(TTOpAtA, true, MALS_projection::RitzGalerkin, TTAtb, TTx, nSweeps, residualTolerance, maxRank, nMALS, nOverlap, nAMEnEnrichment, useTTgmres, gmresMaxIter, gmresRelTol);
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

    // for the Petrov-Galerkin variant, we get a dedicated left projection
    TensorTrain<T> TTw(0,0);
    const TensorTrain<T>& TTv = projection == MALS_projection::PetrovGalerkin ? TTw : TTx;

    // we store previous parts of w^Tb from left and right
    SweepData vTb = defineSweepData<Tensor2<T>>(nDim, dot_loop_from_left<T>(TTv, TTb), dot_loop_from_right<T>(TTv, TTb));
    
    // this includes a calculation of Ax, so allow to store the new parts of Ax in a seperate vector
    std::vector<Tensor3<T>> tmpAx(nDim);
    TensorTrain<T> TTAx(TTOpA.row_dimensions());
    SweepData Ax = defineSweepData<Tensor3<T>>(nDim, apply_loop(TTOpA, TTx), apply_loop(TTOpA, TTx));

    // we store previous parts of x^T A x
    SweepData vTAx = defineSweepData<Tensor2<T>>(nDim, dot_loop_from_left<T>(TTv, Ax), dot_loop_from_right<T>(TTv, Ax));

    // save local residual for enriching the subspace
    std::unique_ptr<TensorTrain<T>> tt_r;

    // calculate the error norm
    apply(TTOpA, TTx, TTAx);
    T residualNorm = axpby(T(1), TTb, T(-1), TTAx, T(0));
    std::cout << "Initial residual norm: " << residualNorm << " (abs), " << residualNorm / nrm_TTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";

    // lambda to avoid code duplication: performs one step in a sweep
    const auto solveLocalProblem = [&](const internal::SweepIndex &swpIdx, bool firstSweep = false)
    {
      std::cout << " (M)ALS setup local problem for sub-tensors " << swpIdx.leftDim() << " to " << swpIdx.rightDim() << "\n";

      internal::ensureLeftOrtho_range(TTx, 0, swpIdx.leftDim());
      internal::ensureRightOrtho_range(TTx, swpIdx.rightDim(), nDim - 1);
      Ax.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);

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
      }

      vTAx.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);
      vTb.update(swpIdx.leftDim()-1, swpIdx.rightDim()+1);

      // prepare operator and right-hand side
      TensorTrain<T> tt_x = calculate_local_x(swpIdx.leftDim(), nMALS, TTx);
      const TensorTrain<T> tt_b = calculate_local_rhs<T>(swpIdx.leftDim(), nMALS, vTb.left(), TTb, vTb.right());
      const TensorTrainOperator<T> localTTOp = calculate_local_op<T>(swpIdx.leftDim(), nMALS, vTAx.left(), TTOpA, vTAx.right());

      assert(check_systemDimensions(localTTOp, tt_x, tt_b));
      assert(check_localProblem(TTOpA, TTx, TTb, TTw, projection == MALS_projection::RitzGalerkin, swpIdx, localTTOp, tt_x, tt_b));

      // first Sweep: let GMRES start from zero, at least favorable for TT-GMRES!
      if (firstSweep && residualNorm / nrm_TTb > 0.5)
        tt_x.setZero();
      
      if (useTTgmres)
        const T localRes = solveGMRES(localTTOp, tt_b, tt_x, gmresMaxIter, gmresRelTol * residualTolerance * nrm_TTb, gmresRelTol, maxRank, true, symmetric, " (M)ALS local problem: ", true);
      else
        const T localRes = solveDenseGMRES(localTTOp, symmetric, tt_b, tt_x, maxRank, gmresMaxIter, gmresRelTol * residualTolerance * nrm_TTb, gmresRelTol, " (M)ALS local problem: ", true);
      
      if( nAMEnEnrichment > 0 )
      {
        tt_r.reset(new TensorTrain<T>(tt_b.dimensions()));
        apply(localTTOp, tt_x, *tt_r);
        axpby(T(1), tt_b, T(-1), *tt_r, T(0));
      }

      TTx.setSubTensors(swpIdx.leftDim(), std::move(tt_x));

#ifndef NDEBUG
if( nAMEnEnrichment > 0 )
{
  TensorTrainOperator<T> TTOpW = projection == MALS_projection::PetrovGalerkin ? setupProjectionOperator(TTw, swpIdx) : setupProjectionOperator(TTx, swpIdx);
  std::vector<int> colDimOpW_left(TTOpW.column_dimensions().size());
  for(int iDim = 0; iDim < nDim; iDim++)
    colDimOpW_left[iDim] = iDim < swpIdx.rightDim() ? TTOpW.column_dimensions()[iDim] : TTOpW.row_dimensions()[iDim];
  TensorTrainOperator<T> TTOpW_left(TTOpW.row_dimensions(), colDimOpW_left);
  TTOpW_left.setEye();
  std::vector<Tensor3<T>> tmpSubT(swpIdx.leftDim());
  for(int iDim = 0; iDim < swpIdx.leftDim(); iDim++)
    copy(TTOpW.tensorTrain().subTensor(iDim), tmpSubT[iDim]);
  TTOpW_left.tensorTrain().setSubTensors(0, std::move(tmpSubT));
  TensorTrain<T> TTz = transpose(TTOpW_left) * (TTb - TTOpA * TTx);
  rightNormalize(TTz);
  tmpSubT.resize(nMALS);
  for(int iDim = swpIdx.leftDim(); iDim <= swpIdx.rightDim(); iDim++)
    copy(TTz.subTensor(iDim), tmpSubT[iDim-swpIdx.leftDim()]);
  tt_r->setSubTensors(0, std::move(tmpSubT));
}
#endif

      if( projection == MALS_projection::PetrovGalerkin )
      {
        // recover original sub-tensor of projection space
        vTb.invalidate(0, nDim);
        vTAx.invalidate(0, nDim);
      }
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

      if( leftToRight )
      {
        std::vector<Tensor3<T>> newSubT(2);
        {
          internal::ensureLeftOrtho_range(TTx, 0, swpIdx.rightDim()+1);
          leftNormalize(*tt_r, T(0));
          const Tensor3<T>& subT0 = TTx.subTensor(swpIdx.rightDim());
          const Tensor3<T>& subTr = tt_r->subTensor(tt_r->dimensions().size()-1);//TTz.subTensor(swpIdx.rightDim());
          const Tensor3<T>& subT1 = TTx.subTensor(swpIdx.rightDim()+1);
          const int addRank = std::min<int>(nAMEnEnrichment, subTr.r2());
          std::cout << " Enhancing subspace (left-to-right) for sub-tensor " << swpIdx.rightDim() << " for optimizing sub-tensor " << swpIdx.rightDim()+1 << ": increasing rank from " << subT0.r2() << " to " << subT0.r2()+addRank << "\n";
          newSubT[0].resize(subT0.r1(), subT0.n(), subT0.r2()+addRank);
          newSubT[1].resize(subT1.r1()+addRank, subT1.n(), subT1.r2());
          newSubT[0].setConstant(T(0));
          newSubT[1].setConstant(T(0));
#pragma omp parallel for collapse(3) schedule(static)
          for(int k = 0; k < subT0.r2(); k++)
            for(int j = 0; j < subT0.n(); j++)
              for(int i = 0; i < subT0.r1(); i++)
                newSubT[0](i,j,k) = subT0(i,j,k);
          const auto r1 = std::min(subT0.r1(), subTr.r1());
#pragma omp parallel for collapse(3) schedule(static)
          for(int k = 0; k < addRank; k++)
            for(int j = 0; j < subT0.n(); j++)
              for(int i = 0; i < r1; i++)
                newSubT[0](i,j,subT0.r2()+k) = subTr(i,j,k);
#pragma omp parallel for collapse(3) schedule(static)
          for(int k = 0; k < subT1.r2(); k++)
            for(int j = 0; j < subT1.n(); j++)
              for(int i = 0; i < subT1.r1(); i++)
                newSubT[1](i,j,k) = subT1(i,j,k);
        }
        TTx.setSubTensors(swpIdx.rightDim(), std::move(newSubT));

        // these are not valid any more
        Ax.invalidate(swpIdx.rightDim()+1);
        vTAx.invalidate(swpIdx.rightDim()+1);
        vTb.invalidate(swpIdx.rightDim()+1);
      }
      else // right-to-left
      {

        std::vector<Tensor3<T>> newSubT(2);
        {
          internal::ensureRightOrtho_range(TTx, swpIdx.leftDim()-1, swpIdx.nDim()-1);
          rightNormalize(*tt_r, T(0));
          const Tensor3<T>& subT0 = TTx.subTensor(swpIdx.leftDim()-1);
          internal::ensureRightOrtho_range(*tt_r, 0, nMALS-1);
          const Tensor3<T>& subTr = tt_r->subTensor(0);//TTz.subTensor(swpIdx.leftDim());
          const Tensor3<T>& subT1 = TTx.subTensor(swpIdx.leftDim());
          const int addRank = std::min<int>(nAMEnEnrichment, subTr.r1());
          std::cout << " Enhancing subspace (right-to-left) for sub-tensor " << swpIdx.leftDim() << " for optimizing sub-tensor " << swpIdx.leftDim()-1 << ": increasing rank from " << subT0.r2() << " to " << subT0.r2()+addRank << "\n";
          newSubT[0].resize(subT0.r1(), subT0.n(), subT0.r2()+addRank);
          newSubT[1].resize(subT1.r1()+addRank, subT1.n(), subT1.r2());
          newSubT[0].setConstant(T(0));
          newSubT[1].setConstant(T(0));
#pragma omp parallel for collapse(3) schedule(static)
          for(int i = 0; i < subT0.r1(); i++)
            for(int j = 0; j < subT0.n(); j++)
              for(int k = 0; k < subT0.r2(); k++)
                newSubT[0](i,j,k) = subT0(i,j,k);
#pragma omp parallel for collapse(3) schedule(static)
          for(int i = 0; i < subT1.r1(); i++)
            for(int j = 0; j < subT1.n(); j++)
              for(int k = 0; k < subT1.r2(); k++)
                newSubT[1](i,j,k) = subT1(i,j,k);
          const auto r2 = std::min(subT1.r2(), subTr.r2());
#pragma omp parallel for collapse(3) schedule(static)
          for(int i = 0; i < addRank; i++)
            for(int j = 0; j < subT0.n(); j++)
              for(int k = 0; k < r2; k++)
                newSubT[1](subT1.r1()+i,j,k) = subTr(i,j,k);
        }
        TTx.setSubTensors(swpIdx.leftDim()-1, std::move(newSubT));

        // these are not valid any more
        Ax.invalidate(swpIdx.leftDim()-1);
        vTAx.invalidate(swpIdx.leftDim()-1);
        vTb.invalidate(swpIdx.leftDim()-1);
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
        if( nMALS == nDim || swpIdx != lastSwpIdx )
          solveLocalProblem(swpIdx, iSweep == 0);

        if( nAMEnEnrichment > 0 )
          enrichSubspace(swpIdx, true);
        
        lastSwpIdx = swpIdx;
      }
      // update remaining sub-tensors of Ax
      Ax.update(nDim-1, nDim);
      for(int iDim = 0; iDim < nDim; iDim++)
        copy(Ax.subTensor(iDim), tmpAx[iDim]);
      tmpAx = TTAx.setSubTensors(0, std::move(tmpAx));



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

        //if( nMALS == 1 && swpIdx.rightDim() < nDim-1 && nAMEnEnrichment > 0 )
        //  enrichSubSpace(swpIdx.next(), false);

        solveLocalProblem(swpIdx);
        lastSwpIdx = swpIdx;
      }
      // update remaining sub-tensors of right_Ax
      Ax.update(-1, 0);
      for(int iDim = 0; iDim < nDim; iDim++)
        copy(Ax.subTensor(iDim), tmpAx[iDim]);
      tmpAx = TTAx.setSubTensors(0, std::move(tmpAx));

      assert(norm2(TTOpA * TTx - TTAx) < sqrt_eps);

      // check error
      residualNorm = axpby(T(1), TTb, T(-1), TTAx, T(0));
      std::cout << "Sweep " << iSweep+1 << " residual norm: " << residualNorm << " (abs), " << residualNorm / nrm_TTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";
    }


    return residualNorm;
  }

}


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_IMPL_HPP
