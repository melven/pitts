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
#include "pitts_tensortrain_solve_mals_helper.hpp"
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
            for(int jChunk = 0; jChunk < subT0.nChunks(); jChunk++)
              for(int i = 0; i < subT0.r1(); i++)
                newSubT[0].chunk(i,jChunk,k) = subT0.chunk(i,jChunk,k);
          const auto r1 = std::min(subT0.r1(), subTr.r1());
#pragma omp parallel for collapse(3) schedule(static)
          for(int k = 0; k < addRank; k++)
            for(int jChunk = 0; jChunk < subT0.nChunks(); jChunk++)
              for(int i = 0; i < r1; i++)
                newSubT[0].chunk(i,jChunk,subT0.r2()+k) = subTr.chunk(i,jChunk,k);
#pragma omp parallel for collapse(3) schedule(static)
          for(int k = 0; k < subT1.r2(); k++)
            for(int jChunk = 0; jChunk < subT1.nChunks(); jChunk++)
              for(int i = 0; i < subT1.r1(); i++)
                newSubT[1].chunk(i,jChunk,k) = subT1.chunk(i,jChunk,k);
        }
        TTx.setSubTensors(swpIdx.rightDim(), std::move(newSubT));

        // these are not valid any more
        right_Ax.pop_back();
        if( !right_vTAx.empty() )
          right_vTAx.pop_back();
        if( !right_vTb.empty() )
          right_vTb.pop_back();
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
            for(int jChunk = 0; jChunk < subT0.n(); jChunk++)
              for(int k = 0; k < subT0.r2(); k++)
                newSubT[0].chunk(i,jChunk,k) = subT0.chunk(i,jChunk,k);
#pragma omp parallel for collapse(3) schedule(static)
          for(int i = 0; i < subT1.r1(); i++)
            for(int jChunk = 0; jChunk < subT1.n(); jChunk++)
              for(int k = 0; k < subT1.r2(); k++)
                newSubT[1].chunk(i,jChunk,k) = subT1.chunk(i,jChunk,k);
          const auto r2 = std::min(subT1.r2(), subTr.r2());
#pragma omp parallel for collapse(3) schedule(static)
          for(int i = 0; i < addRank; i++)
            for(int jChunk = 0; jChunk < subT0.n(); jChunk++)
              for(int k = 0; k < r2; k++)
                newSubT[1].chunk(subT1.r1()+i,jChunk,k) = subTr.chunk(i,jChunk,k);
        }
        TTx.setSubTensors(swpIdx.leftDim()-1, std::move(newSubT));

        // these are not valid any more
        left_Ax.pop_back();
        if( !left_vTAx.empty() )
          left_vTAx.pop_back();
        if( !left_vTb.empty() )
          left_vTb.pop_back();
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

        //if( nMALS == 1 && swpIdx.rightDim() < nDim-1 && nAMEnEnrichment > 0 )
        //  enrichSubSpace(swpIdx.next(), false);

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
      residualNorm = axpby(T(1), TTb, T(-1), TTAx, T(0));
      std::cout << "Sweep " << iSweep+1 << " residual norm: " << residualNorm << " (abs), " << residualNorm / nrm_TTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";
    }


    return residualNorm;
  }

}


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_IMPL_HPP
