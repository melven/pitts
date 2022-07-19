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

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    
    //! dedicated helper functions for solveMALS
    namespace solve_mals
    {
      //! calculate next part of x^Tb from right to left (like TT dot product bug allows to store all intermediate results)
      template<typename T>
      Tensor2<T> calculate_next_right_xTb(const Tensor3<T>& subTb, const Tensor3<T>& subTx, const Tensor2<T>& prev_xTb)
      {
        // first contraction: subTb(:,:,*) * prev_t2(:,*)
        Tensor3<T> t3_tmp;
        internal::dot_contract1(subTb, prev_xTb, t3_tmp);

        // second contraction: subTx(:,*,*) * t3_tmp(:,*,*)
        Tensor2<T> t2;
        internal::dot_contract2(subTx, t3_tmp, t2);
        return t2;
      }

      //! calculate next part of x^Tb from left to right
      template<typename T>
      Tensor2<T> calculate_next_left_xTb(const Tensor3<T>& subTb, const Tensor3<T>& subTx, const Tensor2<T>& prev_xTb)
      {
        // first contraction: pve_t2(*,:) * subTb(*,:,:)
        Tensor3<T> t3_tmp;
        internal::reverse_dot_contract1(prev_xTb, subTb, t3_tmp);

        // second contraction: t3(*,*,:) * subTx(*,*,:)
        Tensor2<T> t2;
        internal::reverse_dot_contract2(t3_tmp, subTx, t2);
        return t2;
      }

      //! calculate next part of x^TAx from right to left
      //! again like TT dot product but with Ax
      //! we have
      //!  |   |     |
      //!  -- xTAx --
      //!
      //! and we need for the next step
      //!   |        |      |
      //!  x_k^T -- A_k -- x_k
      //!   |        |      |
      //!   ------ xTAx -----
      //!
      //! Also calculates the current sub-tensor of A*x
      //!
      template<typename T>
      Tensor2<T> calculate_next_right_xTAx(const TensorTrainOperator<T>& TTOpA, int iDim, const Tensor3<T>& Ak, const Tensor3<T>& xk, const Tensor2<T>& prev_xTAx, Tensor3<T>& Axk)
      {
        // first contract A_k with x_k
        //     |      |
        // -- A_k -- x_k
        //     |      |
        //                    
        //
        internal::apply_contract(TTOpA, iDim, Ak, xk, Axk);
        // now we have
        //   |        ||
        //  x_k^T -- Axk
        //   |        ||
        //   ------ xTAx

        // now contract Axk and xTAx
        //            ||
        //        -- Axk
        //            ||
        //       -- xTAx
        Tensor3<T> t3_tmp;
        internal::dot_contract1(Axk, prev_xTAx, t3_tmp);
        // now we have
        //   |        ||
        //  x_k^T --- t3
        //   |________|
        //
        Tensor2<T> t2;
        internal::dot_contract2(xk, t3_tmp, t2);
        return t2;
      }

      //! calculate next part of x^TAx from left to right
      //!
      //! Also calculates the current sub-tensor of A*x
      //!
      template<typename T>
      Tensor2<T> calculate_next_left_xTAx(const TensorTrainOperator<T>& TTOpA, int iDim, const Tensor3<T>& Ak, const Tensor3<T>& xk, const Tensor2<T>& prev_xTAx, Tensor3<T>& Axk)
      {
        internal::apply_contract(TTOpA, iDim, Ak, xk, Axk);
        // now we have
        //   ------ xTAx
        //   |        ||
        //  x_k^T -- Axk
        //   |        ||

        // now contract Axk and xTAx
        //       -- xTAx
        //            ||
        //        -- Axk
        //            ||
        Tensor3<T> t3_tmp;
        // first contraction: xTAx(*,:) * Axk(*,:,:)
        internal::reverse_dot_contract1(prev_xTAx, Axk, t3_tmp);
        // now we have
        //   __________
        //   |        |
        //  x_k^T --- t3
        //   |        ||
        //
        Tensor2<T> t2;
        // second contraction: t3_tmp(*,*,:) * xk(*,*,:)
        internal::reverse_dot_contract2(t3_tmp, xk, t2);
        return t2;
      }

      //! calculate the local RHS tensor-train for (M)ALS
      template<typename T>
      TensorTrain<T> calculate_local_rhs(int iDim, int nMALS, const Tensor2<T>& left_xTb, const TensorTrain<T>& TTb, const Tensor2<T>& right_xTb)
      {
        std::vector<int> dims(nMALS);
        copy_n(TTb.dimensions().begin() + iDim, nMALS, dims.begin());
  
        TensorTrain<T> tt_b(dims);
        for(int i = 0; i < nMALS; i++)
          copy(TTb.subTensors()[iDim+i], tt_b.editableSubTensors()[i]);

        // first contract: tt_b_right(:,:,*) * right_xTb(:,*)
        Tensor3<T> t3_tmp;
        std::swap(tt_b.editableSubTensors().back(), t3_tmp);
        internal::dot_contract1(t3_tmp, right_xTb, tt_b.editableSubTensors().back());

        // then contract: left_xTb(*,:) * tt_b_left(*,:,:)
        std::swap(tt_b.editableSubTensors().front(), t3_tmp);
        internal::reverse_dot_contract1(left_xTb, t3_tmp, tt_b.editableSubTensors().front());

        return tt_b;
      }

      //! calculate the local initial solutiuon in TT format for (M)ALS
      template<typename T>
      TensorTrain<T> calculate_local_x(int iDim, int nMALS, const TensorTrain<T>& TTx)
      {
        std::vector<int> dims(nMALS);
        copy_n(TTx.dimensions().begin() + iDim, nMALS, dims.begin());
  
        TensorTrain<T> tt_x(dims);
        for(int i = 0; i < nMALS; i++)
          copy(TTx.subTensors()[iDim+i], tt_x.editableSubTensors()[i]);

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
      TensorTrainOperator<T> calculate_local_op(int iDim, int nMALS, const Tensor2<T>& left_xTAx, const TensorTrainOperator<T>& TTOp, const Tensor2<T>& right_xTAx)
      {
        const int n0 = left_xTAx.r2();
        const int nd = right_xTAx.r1();

        std::vector<int> localRowDims(nMALS+2), localColDims(nMALS+2);
        localRowDims.front() = localColDims.front() = n0;
        for(int i = 0; i < nMALS; i++)
        {
          localRowDims[i+1] = TTOp.row_dimensions()[iDim+i];
          localColDims[i+1] = TTOp.column_dimensions()[iDim+i];
        }
        localRowDims.back()  = localColDims.back()  = nd;

        TensorTrainOperator<T> localTTOp(localRowDims, localColDims);
        copy_op_left(left_xTAx, localTTOp.tensorTrain().editableSubTensors().front());
        for(int i = 0; i < nMALS; i++)
          copy(TTOp.tensorTrain().subTensors()[iDim+i], localTTOp.tensorTrain().editableSubTensors()[1+i]);
        copy_op_right(right_xTAx, localTTOp.tensorTrain().editableSubTensors().back());

        return localTTOp;
      }
    }

  }


  //! Solve a linear system using the MALS algorithm
  //!
  //! Approximate x with Ax = b
  //!
  //! @tparam T             data type (double, float, complex)
  //!
  //! @param TTOpA              tensor-train operator A
  //! @param symmetricA         flag to indicate that A is symmetric / Hermitian
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
              const bool symmetricA,
              const TensorTrain<T>& TTb,
              TensorTrain<T>& TTx,
              int nSweeps,
              T residualTolerance = std::sqrt(std::numeric_limits<T>::epsilon()),
              int maxRank = std::numeric_limits<int>::max(),
              int nMALS = 2, int nOverlap = 1,
              bool useTTgmres = false, int gmresMaxIter = 25, T gmresRelTol = T(1.e-4))
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using arr = Eigen::Array<T, 1, Eigen::Dynamic>;
    using namespace internal::solve_mals;

    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // check that dimensions match
    if( TTb.dimensions() != TTOpA.row_dimensions() )
      throw std::invalid_argument("TensorTrain solveMALS: operator and rhs dimensions mismatch!");
    if( TTx.dimensions() != TTOpA.column_dimensions() )
      throw std::invalid_argument("TensorTrain solveMALS: operator and x dimensions mismatch!");

    const int nDim = TTx.subTensors().size();
    nMALS = std::min(nMALS, nDim);
    nOverlap = std::min(nOverlap, nMALS-1);
    if( nMALS < 1 || nMALS > nDim )
      throw std::invalid_argument("Tensortrain solveMALS: invalid parameter nMALS (1 <= nMALS)!");
    if( nOverlap < 0 || nOverlap >= nMALS )
      throw std::invalid_argument("Tensortrain solveMALS: invalid parameter nOverlap (1 <= nOverlap < nMALS)!");

    // first left-normalize x
    rightNormalize(TTx, residualTolerance/nDim, maxRank);


    // for the non-symmetric case, we solve the normal equations, so calculate A^T*b and A^T*A
    TensorTrain<T> TTAtb(TTOpA.column_dimensions());
    TensorTrainOperator<T> TTOpAtA(TTOpA.column_dimensions(), TTOpA.column_dimensions());
    if( !symmetricA )
    {
      applyT(TTOpA, TTb, TTAtb);
      applyT(TTOpA, TTOpA, TTOpAtA);
    }

    // provide convenient name for TTOpA resp. TTOpAtA for the code below
    const auto& effTTb = symmetricA ? TTb : TTAtb;
    const auto effTTOpA = symmetricA ? TTOpA : TTOpAtA;

    const T bTb = dot(effTTb,effTTb);
    const T sqrt_bTb = std::sqrt(bTb);

#ifndef NDEBUG
    constexpr auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());
#endif

    // helper function that returns a 1x1 Tensor2 with value 1
    constexpr auto Tensor2_one = []()
    {
      Tensor2<T> t2(1,1);
      t2(0,0) = T(1);
      return t2;
    };


    // we store previous parts of x^Tb from left and right
    // (respectively x^T A^T b for the non-symmetric case)
    std::vector<Tensor2<T>> left_xTb, right_xTb;
    left_xTb.emplace_back(Tensor2_one());
    right_xTb.emplace_back(Tensor2_one());
    
    // like TT dot product and store all intermediate results
    for(int iDim = nDim-1; iDim >= 0; iDim--)
      right_xTb.emplace_back( calculate_next_right_xTb(effTTb.subTensors()[iDim], TTx.subTensors()[iDim], right_xTb.back()) );

    assert(right_xTb.size() == nDim+1);
    assert(right_xTb[nDim].r1() == 1 && right_xTb[nDim].r2() == 1);
    assert( std::abs(right_xTb[nDim](0,0) - dot(TTx, effTTb)) < sqrt_eps );


    // we store previous parts of x^T A x
    // (respectively x^T A^T A x for the non-symmetric case)
    std::vector<Tensor2<T>> left_xTAx, right_xTAx;
    left_xTAx.emplace_back(Tensor2_one());
    right_xTAx.emplace_back(Tensor2_one());

    // this includes a calculation of Ax, so reuse it
    TensorTrain<T> TTAx(effTTOpA.row_dimensions()), residualVector(effTTOpA.row_dimensions());
    for(int iDim = nDim-1; iDim >= 0; iDim--)
      right_xTAx.emplace_back( calculate_next_right_xTAx(effTTOpA, iDim, effTTOpA.tensorTrain().subTensors()[iDim], TTx.subTensors()[iDim], right_xTAx.back(), TTAx.editableSubTensors()[iDim]) );

    assert(right_xTAx.size() == nDim+1);
    assert(right_xTAx[nDim].r1() == 1 && right_xTAx[nDim].r2() == 1);
    assert( std::abs( right_xTAx[nDim](0,0) - dot(TTx, TTAx) ) < sqrt_eps );
#ifndef NDEBUG
    constexpr auto apply_error = [](const TensorTrainOperator<T>& A, const TensorTrain<T>& x, const TensorTrain<T>& Ax) -> T
    {
      TensorTrain<T> Ax_ref(A.row_dimensions());
      apply(A, x, Ax_ref);
      return std::abs(axpby(T(1), Ax, T(-1), Ax_ref));
    };
#endif
    assert( apply_error(effTTOpA, TTx, TTAx) < sqrt_eps );


    // calculate the error norm
    copy(effTTb, residualVector);
    auto residualNorm = axpby(T(-1),TTAx,T(1),residualVector);
    std::cout << "Initial residual norm: " << residualNorm << " (abs), " << residualNorm / sqrt_bTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";

    // now everything is prepared, perform the sweeps
    for(int iSweep = 0; iSweep < nSweeps; iSweep++)
    {
      if( residualNorm / sqrt_bTb < residualTolerance )
        break;

      for(int iDim = 1; iDim < nMALS; iDim++)
      {
        right_xTb.pop_back();
        right_xTAx.pop_back();
      }

      // sweep left to right
      for(int iDim = 0; iDim < nDim; iDim++)
      {
        if( iDim + nMALS <= nDim )
        {
          right_xTb.pop_back();
          right_xTAx.pop_back();
        }

        if( iDim + nMALS <= nDim && iDim % (nMALS-nOverlap) == 0 )
        {
          std::cout << " (M)ALS setup local problem for sub-tensors " << iDim << " to " << iDim+nMALS-1 << "\n";

          // prepare operator and right-hand side
          TensorTrain<T> tt_x = calculate_local_x(iDim, nMALS, TTx);
          const TensorTrain<T> tt_b = calculate_local_rhs(iDim, nMALS, left_xTb.back(), effTTb, right_xTb.back());
          const TensorTrainOperator<T> localTTOp = calculate_local_op(iDim, nMALS, left_xTAx.back(), effTTOpA, right_xTAx.back());
          assert( std::abs( dot(tt_x, tt_b) - dot(TTx, effTTb) ) < sqrt_eps );

          if( useTTgmres )
          {
            const T localRes = solveGMRES(localTTOp, tt_b, tt_x, gmresMaxIter, T(0.), gmresRelTol, maxRank, true, " (M)ALS local problem: ", true);
          }
          else
          {
            // GMRES with dense vectors...
            MultiVector<T> mv_x, mv_rhs;
            toDense(tt_x, mv_x);
            toDense(tt_b, mv_rhs);

            // absolute tolerance is not invariant wrt. #dimensions
            const auto localRes = GMRES<arr>(localTTOp, true, mv_rhs, mv_x, gmresMaxIter, arr::Constant(1, 0), arr::Constant(1, gmresRelTol), " (M)ALS local problem: ", true);

            // use mv_rhs as work array
            const auto r_left = tt_x.subTensors().front().r1();
            const auto r_right = tt_x.subTensors().back().r2();
            TensorTrain<T> new_tt_x = fromDense(mv_x, mv_rhs, tt_x.dimensions(), T(0.5)*residualTolerance/nDim, maxRank, false, r_left, r_right);
            std::swap(tt_x, new_tt_x);
          }

          for (int i = 0; i < tt_x.dimensions().size(); i++)
            std::swap(tt_x.editableSubTensors()[i], TTx.editableSubTensors()[iDim + i]);
          if (iDim + 1 != nDim)
            internal::leftNormalize_range(TTx, iDim, iDim + 1);
        }

        // prepare left/right xTb for the next iteration
        left_xTb.emplace_back( calculate_next_left_xTb(effTTb.subTensors()[iDim], TTx.subTensors()[iDim], left_xTb.back()) );
        left_xTAx.emplace_back( calculate_next_left_xTAx(effTTOpA, iDim, effTTOpA.tensorTrain().subTensors()[iDim], TTx.subTensors()[iDim], left_xTAx.back(), TTAx.editableSubTensors()[iDim]) );

        if( iDim+1 < nDim )
        {
          const auto& Ak_next = effTTOpA.tensorTrain().subTensors()[iDim+1];
          const auto& xk_next = TTx.subTensors()[iDim+1];
          auto& Axk_next = TTAx.editableSubTensors()[iDim+1];
          internal::apply_contract(effTTOpA, iDim+1, Ak_next, xk_next, Axk_next);
        }

        assert( apply_error(effTTOpA, TTx, TTAx) < sqrt_eps );
      }
      assert(left_xTb.size() == nDim+1);
      assert(left_xTb[nDim].r1() == 1 && left_xTb[nDim].r2() == 1);
      assert( std::abs( left_xTb[nDim](0,0) - dot(TTx, effTTb) ) < sqrt_eps );
      assert(left_xTAx.size() == nDim+1);
      assert(left_xTAx[nDim].r1() == 1 && left_xTAx[nDim].r2() == 1);
      assert( std::abs( left_xTAx[nDim](0,0) - dot(TTx,TTAx) ) < sqrt_eps );
      assert(right_xTb.size() == 1);
      assert(right_xTAx.size() == 1);


      // check error
      copy(effTTb, residualVector);
      residualNorm = axpby(T(-1),TTAx,T(1),residualVector);
      std::cout << "Sweep " << iSweep+0.5 << " residual norm: " << residualNorm << " (abs), " << residualNorm / sqrt_bTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";
      if( residualNorm / sqrt_bTb < residualTolerance )
        break;

      for(int iDim = nDim-1; iDim > nDim-nMALS; iDim--)
      {
        left_xTb.pop_back();
        left_xTAx.pop_back();
      }

      // sweep right to left
      for(int iDim = nDim-1; iDim >= 0; iDim--)
      {
        if( iDim+1 - nMALS >= 0 )
        {
          left_xTb.pop_back();
          left_xTAx.pop_back();
        }

        if( iDim+1 - nMALS >= 0 && (nDim-iDim-1) % (nMALS-nOverlap) == 0 )
        {
          std::cout << " (M)ALS setup local problem for sub-tensors " << iDim << " to " << iDim+1-nMALS << "\n";

          // prepare operator and right-hand side
          TensorTrain<T> tt_x = calculate_local_x(iDim+1-nMALS, nMALS, TTx);
          const TensorTrain<T> tt_b = calculate_local_rhs(iDim+1-nMALS, nMALS, left_xTb.back(), effTTb, right_xTb.back());
          const TensorTrainOperator<T> localTTOp = calculate_local_op(iDim+1-nMALS, nMALS, left_xTAx.back(), effTTOpA, right_xTAx.back());
          assert( std::abs( dot(tt_x, tt_b) - dot(TTx, effTTb) ) < sqrt_eps );

          if( useTTgmres )
          {
            const T localRes = solveGMRES(localTTOp, tt_b, tt_x, gmresMaxIter, T(0.), gmresRelTol, maxRank, true, " (M)ALS local problem: ", true);
          }
          else
          {
            // GMRES with dense vectors...
            MultiVector<T> mv_x, mv_rhs;
            toDense(tt_x, mv_x);
            toDense(tt_b, mv_rhs);

            // absolute tolerance is not invariant wrt. #dimensions
            const auto localRes = GMRES<arr>(localTTOp, true, mv_rhs, mv_x, gmresMaxIter, arr::Constant(1, 0), arr::Constant(1, gmresRelTol), " (M)ALS local problem: ", true);

            const auto r_left = tt_x.subTensors().front().r1();
            const auto r_right = tt_x.subTensors().back().r2();
            TensorTrain<T> new_tt_x = fromDense(mv_x, mv_rhs, tt_x.dimensions(), T(0.5)*residualTolerance/nDim, maxRank, false, r_left, r_right);
            std::swap(tt_x, new_tt_x);
          }

          for(int i = 0; i < nMALS; i++)
            std::swap(tt_x.editableSubTensors()[i], TTx.editableSubTensors()[iDim+i+1-nMALS]);
          if( iDim != 0 )
            internal::rightNormalize_range(TTx, iDim-1, iDim);
        }

        // prepare left/right xTb for the next iteration
        right_xTb.emplace_back( calculate_next_right_xTb(effTTb.subTensors()[iDim], TTx.subTensors()[iDim], right_xTb.back()) );
        right_xTAx.emplace_back( calculate_next_right_xTAx(effTTOpA, iDim, effTTOpA.tensorTrain().subTensors()[iDim], TTx.subTensors()[iDim], right_xTAx.back(), TTAx.editableSubTensors()[iDim]) );

        if( iDim > 0 )
        {
          const auto& Ak_prev = effTTOpA.tensorTrain().subTensors()[iDim-1];
          const auto& xk_prev = TTx.subTensors()[iDim-1];
          auto& Axk_prev = TTAx.editableSubTensors()[iDim-1];
          internal::apply_contract(effTTOpA, iDim-1, Ak_prev, xk_prev, Axk_prev);
        }

        assert( apply_error(effTTOpA, TTx, TTAx) < sqrt_eps );
      }
      assert(right_xTb.size() == nDim+1);
      assert(right_xTb[nDim].r1() == 1 && right_xTb[nDim].r2() == 1);
      assert( std::abs( right_xTb[nDim](0,0) - dot(TTx, effTTb) ) < sqrt_eps );
      assert(right_xTAx.size() == nDim+1);
      assert(right_xTAx[nDim].r1() == 1 && right_xTAx[nDim].r2() == 1);
      assert( std::abs( right_xTAx[nDim](0,0) - dot(TTx,TTAx) ) < sqrt_eps );
      assert(left_xTb.size() == 1);
      assert(left_xTAx.size() == 1);


      // check error
      copy(effTTb, residualVector);
      residualNorm = axpby(T(-1),TTAx,T(1),residualVector);
      std::cout << "Sweep " << iSweep+1 << " residual norm: " << residualNorm << " (abs), " << residualNorm / sqrt_bTb << " (rel), ranks: " << internal::to_string(TTx.getTTranks()) << "\n";
    }


    return residualNorm;
  }

}


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_HPP
