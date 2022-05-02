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
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_operator_apply_transposed.hpp"
#include "pitts_tensortrain_operator_apply_transposed_op.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor3 and Tensor2 along first dimensions: A(*,:) * B(*,:,:)
    template<typename T>
    void reverse_dot_contract1(const Tensor2<T>& A, const Tensor3<T>& B, Tensor3<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = B.n();
      const auto nChunks = B.nChunks();
      const auto r2 = B.r2();
      assert(A.r1() == B.r1());
      const auto r1_ = A.r2();
      C.resize(r1_, n, r2);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "nChunks", "r2", "r1_"},{r1, nChunks, r2, r1_}}, // arguments
        {{r1*nChunks*r2*r1_*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1*nChunks*r2+r1*r1_)*kernel_info::Load<Chunk<T>>() + (r1_*nChunks*r2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      for(int jChunk = 0; jChunk < nChunks; jChunk++)
      {
        for(int i = 0; i < r1_; i++)
          for(int k = 0; k < r2; k++)
          {
            Chunk<T> tmp{};
            for(int l = 0; l < r1; l++)
              fmadd(A(l,i), B.chunk(l,jChunk,k), tmp);
            C.chunk(i,jChunk,k) = tmp;
          }
      }
    }

    //! dot product between two Tensor3 (for checking correctness)
    template<typename T>
    T t3_dot(const Tensor3<T>& A, const Tensor3<T>& B)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto r2 = A.r2();
      assert(A.r1() == B.r1());
      assert(A.n() == B.n());
      assert(A.r2() == B.r2());

      T result{};
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2; k++)
            result += A(i,j,k) * B(i,j,k);

      return result;
    }

    //! contract Tensor3 and Tensor3 along the first two dimensions: A(*,*,:) * B(*,*,:)
    template<typename T>
    void reverse_dot_contract2(const Tensor3<T>& A, const Tensor3<T>& B, Tensor2<T>& C)
    {
      const auto r1 = A.r1();
      const auto n = A.n();
      const auto nChunks = A.nChunks();
      const auto rA2 = A.r2();
      assert(A.r1() == B.r1());
      assert(A.n() == B.n());
      const auto rB2 = B.r2();
      C.resize(rA2,rB2);

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "nChunks", "rA2", "rB2"},{r1, nChunks, rA2, rB2}}, // arguments
        {{r1*nChunks*rA2*rB2*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1*nChunks*rA2+r1*nChunks*rB2)*kernel_info::Load<Chunk<T>>() + (rA2*rB2)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );

      for(int i = 0; i < rA2; i++)
        for(int j = 0; j < rB2; j++)
        {
          T tmp = T(0);
          for(int k = 0; k < r1; k++)
            for(int l = 0; l < n; l++)
              tmp += A(k,l,i) * B(k,l,j);
          C(i,j) = tmp;
        }
    }

  }


  //! Solve a linear system using the MALS algorithm
  //!
  //! Approximate x with Ax = b
  //!
  //! @tparam T             data type (double, float, complex)
  //! @tparam MALS          set to zero to use ALS, and one to use MALS
  //!
  //! @param TTOpA          tensor-train operator A
  //! @param symmetricA     flag to indicate that A is symmetric / Hermitian
  //! @param TTb            right-hand side tensor-train b
  //! @param TTx            initial guess on input, overwritten with the (approximate) result on output
  //! @param nSweeps        desired number of MALS sweeps
  //! @param rankTolerance  approximation accuracy, used to reduce the TTranks in the iteration
  //! @param maxRank        maximal allowed TT-rank, enforced even if this violates the rankTolerance
  //! @return               residual norm of the result (||Ax - b||)
  //!
  template<typename T, bool MALS = 1>
  T solveMALS(const TensorTrainOperator<T>& TTOpA,
              const bool symmetricA,
              const TensorTrain<T>& TTb,
              TensorTrain<T>& TTx,
              int nSweeps,
              T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()),
              int maxRank = std::numeric_limits<int>::max()) 
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // check that dimensions match
    if( TTb.dimensions() != TTOpA.row_dimensions() )
      throw std::invalid_argument("TensorTrain solveMALS dimension mismatch!");
    if( TTx.dimensions() != TTOpA.column_dimensions() )
      throw std::invalid_argument("TensorTrain solveMALS dimension mismatch!");

    const int nDim = TTx.subTensors().size();
    std::cout << "Warning: falling back to ALS for now\n";
    //const bool useMALS = nDim > 1 && MALS;
    const bool useMALS = 0;

    // first left-normalize x
    rightNormalize(TTx, rankTolerance, maxRank);


    // for the non-symmetric case, we solve the normal equations, so calculate A^T*b and A^T*A
    TensorTrain<T> TTAtb(TTOpA.column_dimensions());
    TensorTrainOperator<T> TTOpAtA(TTOpA.column_dimensions(), TTOpA.column_dimensions());
    if( !symmetricA )
    {
      applyT(TTOpA, TTb, TTAtb);
      applyT(TTOpA, TTOpA, TTOpAtA);
    }
    const T bTb = symmetricA ? dot(TTb,TTb) : dot(TTAtb,TTAtb);


    // we store previous parts of x^Tb from left and right
    // (respectively x^T A^T b for the non-symmetric case)
    std::vector<Tensor2<T>> left_xTb(1);
    left_xTb[0].resize(1,1);
    left_xTb[0](0,0) = T(1);
    std::vector<Tensor2<T>> right_xTb(1);
    right_xTb[0].resize(1,1);
    right_xTb[0](0,0) = T(1);
    // like TT dot product and store all intermediate results
    {
      Tensor3<T> t3;
      for(int iDim = nDim-1; iDim >= 0; iDim--)
      {
        const auto& subTb = symmetricA ? TTb.subTensors()[iDim] : TTAtb.subTensors()[iDim];
        const auto& subTx = TTx.subTensors()[iDim];

        // first contraction: subT1(:,:,*) * t2(:,*)
        internal::dot_contract1(subTb, right_xTb.back(), t3);

        // second contraction: subT2(:,*,*) * t3(:,*,*)
        Tensor2<T> t2;
        internal::dot_contract2(subTx, t3, t2);
        right_xTb.emplace_back(std::move(t2));
      }
    }
#ifndef NDEBUG
    {
      assert(right_xTb.size() == nDim+1);
      assert(right_xTb[nDim].r1() == 1);
      assert(right_xTb[nDim].r2() == 1);
      if( symmetricA )
      {
        const auto dot_ref = dot(TTx, TTb);
        const auto err = std::abs(right_xTb[nDim](0,0) - dot_ref);
        assert(err < std::sqrt(std::numeric_limits<T>::epsilon()));
      }
      else
      {
        const auto dot_ref = dot(TTx, TTAtb);
        const auto err = std::abs(right_xTb[nDim](0,0) - dot_ref);
        assert(err < std::sqrt(std::numeric_limits<T>::epsilon()));
      }
    }
#endif


    // we store previous parts of x^T A x
    // (respectively x^T A^T A x for the non-symmetric case)
    std::vector<Tensor2<T>> left_xTAx(1);
    left_xTAx[0].resize(1,1);
    left_xTAx[0](0,0) = T(1);
    std::vector<Tensor2<T>> right_xTAx(1);
    right_xTAx[0].resize(1,1);
    right_xTAx[0](0,0) = T(1);
    // this includes a calculation of Ax, so reuse it
    TensorTrain<T> TTAx(symmetricA ? TTOpA.row_dimensions() : TTOpAtA.row_dimensions());
    // again like TT dot product but with Ax
    {
      // we have
      //  |   |     |
      //  -- xTAx --
      //
      // and we need for the next step
      //   |        |      |
      //  x_k^T -- A_k -- x_k
      //   |        |      |
      //   ------ xTAx -----
      //

      for(int iDim = nDim-1; iDim >= 0; iDim--)
      {
        // first contract A_k with x_k
        //     |      |
        // -- A_k -- x_k
        //     |      |
        //                    
        //
        const auto& TTOp = symmetricA ? TTOpA : TTOpAtA;
        const auto& subTOp = TTOp.tensorTrain().subTensors()[iDim];
        const auto& subTx = TTx.subTensors()[iDim];
        auto& Axk = TTAx.editableSubTensors()[iDim];
        internal::apply_contract(TTOp, iDim, subTOp, subTx, Axk);
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
        Tensor3<T> t3;
        internal::dot_contract1(Axk, right_xTAx.back(), t3);
        // now we have
        //   |        ||
        //  x_k^T --- t3
        //   |________|
        //
        Tensor2<T> next_xTAx;
        internal::dot_contract2(subTx, t3, next_xTAx);

        right_xTAx.emplace_back(std::move(next_xTAx));
      }
    }
#ifndef NDEBUG
    {
      assert(right_xTAx.size() == nDim+1);
      assert(right_xTAx[nDim].r1() == 1);
      assert(right_xTAx[nDim].r2() == 1);
      if( symmetricA )
      {
        TensorTrain<T> TTAx_ref(TTOpA.row_dimensions());
        apply(TTOpA, TTx, TTAx_ref);
        const auto errAx = std::abs(axpby(1., TTAx, -1., TTAx_ref));
        assert(errAx < std::sqrt(std::numeric_limits<T>::epsilon()));
      }
      else
      {
        TensorTrain<T> TTAtAx_ref(TTOpAtA.row_dimensions());
        apply(TTOpAtA, TTx, TTAtAx_ref);
        const auto errAx = std::abs(axpby(1., TTAx, -1., TTAtAx_ref));
        assert(errAx < std::sqrt(std::numeric_limits<T>::epsilon()));
      }
      const auto dot_ref = dot(TTx, TTAx);
      const auto err = std::abs(right_xTAx[nDim](0,0) - dot_ref);
      assert(err < std::sqrt(std::numeric_limits<T>::epsilon()));
    }
#endif


    // calculate the error norm
    // ||Ax - b||_2^2 = <Ax - b, Ax - b> = <Ax,Ax> - 2<Ax, b> + <b,b>
    T squaredError;
    if( symmetricA )
      squaredError = bTb + dot(TTAx,TTAx) - 2*dot(TTAx,TTb);
    else
      squaredError = bTb + dot(TTAx,TTAx) - 2*dot(TTAx,TTAtb);
    auto residualError = std::sqrt(std::abs(squaredError));
    if( symmetricA )
      std::cout << "Initial residual norm: " << residualError << "\n";
    else
      std::cout << "Initial residual norm of the normal equations: " << residualError << "\n";

    // now everything is prepared, perform the sweeps
    for(int iSweep = 0; iSweep < nSweeps; iSweep++)
    {

      // sweep left to right
      for(int iDim = 0; iDim < nDim; iDim++)
      {
        right_xTb.pop_back();
        right_xTAx.pop_back();

        if( iDim + useMALS < nDim )
        {
          // calculate sub-problem right-hand side
          Tensor3<T> t3_rhs;
          {
            const auto& subTb = symmetricA ? TTb.subTensors()[iDim] : TTAtb.subTensors()[iDim];
            Tensor3<T> t3_tmp;
            // first contract: subTb(:,:,*) * t2(:,*)
            internal::dot_contract1(subTb, right_xTb.back(), t3_tmp);
            // then contract: t2(*,:) * t3_tmp(*,:,:)
            internal::reverse_dot_contract1(left_xTb.back(), t3_tmp, t3_rhs);
          }
#ifndef NDEBUG
          {
            // check local RHS
            const auto& subTx = TTx.subTensors()[iDim];
            // contraction t3_rhs and subTx should give xTb, respectively xTAtb
            const auto xTb = internal::t3_dot(subTx, t3_rhs);
            const auto xTb_ref = symmetricA ? dot(TTx, TTb) : dot(TTx, TTAtb);
            const auto xTb_err = std::abs(xTb - xTb_ref);
            assert(xTb_err < std::sqrt(std::numeric_limits<T>::epsilon()));
          }
#endif

          // calculate sub-problem operator
          // --- left_xTAx ---
          //        |
          //   --  A_k --
          //        |
          // --- right_xTAx ---
          Tensor2<T> t2_op;
          {
            const auto& TTOp = symmetricA ? TTOpA : TTOpAtA;
            const auto& Ak = TTOp.tensorTrain().subTensors()[iDim];
            const auto& lOp = left_xTAx.back();
            const auto& rOp = right_xTAx.back();
            Tensor3<T> t3_tmp;
            {
              const auto r1 = lOp.r1() / Ak.r1();
              const auto n = TTOp.row_dimensions()[iDim];
              const auto m = TTOp.column_dimensions()[iDim];
              const auto r2 = lOp.r2();
              const auto rA1 = Ak.r1();
              const auto rA2 = Ak.r2();
//std::cout << "lOp.r1(): " << lOp.r1() << ", lOp.r2(): " << lOp.r2() << "\n";
//std::cout << "   " << r1 << " x " << rA1 << " x " << r2 << "\n";
// check symmetry
/*
{
  for(int i1 = 0; i1 < r1; i1++)
    for(int k1 = 0; k1 < r2; k1++)
      for(int l = 0; l < rA1; l++)
        assert(lOp(i1+l*r1,k1) == lOp(k1+l*r1,i1));
}
*/
              t3_tmp.resize(r1*n, rA2, r2*m);
              for(int i1 = 0; i1 < r1; i1++)
                for(int i2 = 0; i2 < n; i2++)
                  for(int j = 0; j < rA2; j++)
                    for(int k1 = 0; k1 < r2; k1++)
                      for(int k2 = 0; k2 < m; k2++)
                      {
                        T tmp = T(0);
                        for(int l = 0; l < rA1; l++)
                          tmp += lOp(i1+l*r1,k1) * Ak(l,TTOp.index(iDim,i2,k2),j);
                        t3_tmp(k1 + i2*r1, j, i1 + k2*r2) = tmp;
                      }
            }
            // no contract
            // --- t3_tmp ---
            //        |
            // --- right_xTAx ---
            {
              const auto nr1 = t3_tmp.r1();
              const auto rA2 = t3_tmp.n();
              const auto mr2 = t3_tmp.r2();
              const auto r1 = rOp.r1();
              const auto r2 = rOp.r2() / rA2;
/*
{
  std::cout << "rOp.r1(): " << rOp.r1() << ", rOp.r2(): " << rOp.r2() << "\n";
  std::cout << "rOp: " << r1 << " x " << rA2 << " x " << r2 << "\n";
  for(int k = 0; k < rA2; k++)
  {
    for(int i2 = 0; i2 < r1; i2++)
    {
      for(int j2 = 0; j2 < r2; j2++)
        std::cout << " " << rOp(i2,j2+k*r2);
      std::cout << "\n";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
  for(int i2 = 0; i2 < r1; i2++)
    for(int j2 = 0; j2 < r2; j2++)
      for(int k = 0; k < rA2; k++)
        assert( std::abs(rOp(i2+k*r1,j2) - rOp(j2+k*r1,i2)) < std::sqrt(std::numeric_limits<T>::epsilon())); 
}
*/
              t2_op.resize(r1*nr1, r2*mr2);
              for(int i1 = 0; i1 < nr1; i1++)
                for(int i2 = 0; i2 < r1; i2++)
                  for(int j1 = 0; j1 < mr2; j1++)
                    for(int j2 = 0; j2 < r2; j2++)
                    {
                      T tmp = T(0);
                      for(int k = 0; k < rA2; k++)
                        tmp += rOp(i2,j2+k*r2) * t3_tmp(i1,k,j1);
                      t2_op(i1+i2*nr1, j1+j2*mr2) = tmp;
                    }
            }
          }
#ifndef NDEBUG
          {
            // check local operator
            const auto& subTx = TTx.subTensors()[iDim];
            const auto r1 = subTx.r1();
            const auto n = subTx.n();
            const auto r2 = subTx.r2();
            Eigen::Matrix<T, Eigen::Dynamic, 1> x_(r1*n*r2);
            for(int i = 0; i < r1; i++)
              for(int j = 0; j < n; j++)
                for(int k = 0; k < r2; k++)
                  x_(i + j*r1 + k*n*r1) = subTx(i,j,k);
            assert(x_.size() == t2_op.r1());
            assert(x_.size() == t2_op.r2());
            const T xTAx = x_.transpose() * ConstEigenMap(t2_op) * x_;
            const auto xTAx_ref = dot(TTx, TTAx);
            const auto xTAx_err = std::abs(xTAx - xTAx_ref);
            std::cout << "iSweep: " << iSweep << ", iDim: " << iDim << ", xTAx_err: " << xTAx_err << ", xTAx_ref: " << xTAx_ref << ", xTAx: " << xTAx << "\n";
            //Eigen::Matrix<T, Eigen::Dynamic, 1> b_(r1*n*r2);
            //assert(t3_rhs.r1() == r1);
            //assert(t3_rhs.n() == n);
            //assert(t3_rhs.r2() == r2);
            //for(int i = 0; i < r1; i++)
            //  for(int j = 0; j < n; j++)
            //    for(int k = 0; k < r2; k++)
            //      b_(i + j*r1 + k*n*r1) = t3_rhs(i,j,k);
            //std::cout << "  local rhs:\n" << b_.transpose() << "\n";
            //std::cout << "  local x:\n" << x_.transpose() << "\n";
            //std::cout << "  local operator:\n" << ConstEigenMap(t2_op) << std::endl;
            //std::cout << "  symmetry error: " << (ConstEigenMap(t2_op) - ConstEigenMap(t2_op).transpose()).norm() << "\n";
            /*
            TensorTrain<T> TTp(TTx.dimensions()), TTq(TTx.dimensions());
            copy(TTx, TTp);
            copy(TTx, TTq);
            const auto& TTOp = symmetricA ? TTOpA : TTOpAtA;
            TensorTrain<T> TTAq(TTAx.dimensions());
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> t2_op_ref(t2_op.r1(), t2_op.r2());
            for(int i = 0; i < t2_op.r1(); i++)
            {
              for(int j = 0; j < t2_op.r2(); j++)
              {
                int i1 = i % r1;
                int i2 = ( i / r1 ) % n;
                int i3 = i / (n*r1);
                TTp.editableSubTensors()[iDim].setUnit(i1, i2, i3);
                int j1 = j % r1;
                int j2 = ( j / r1 ) % n;
                int j3 = j / (n*r1);
                TTq.editableSubTensors()[iDim].setUnit(j1, j2, j3);
                apply(TTOp, TTq, TTAq);
                t2_op_ref(i,j) = dot(TTp, TTAq);
              }
            }
            std::cout << "  local operator (ref):\n" << t2_op_ref << "\n";
            std::cout << "  local operator error:\n" << ConstEigenMap(t2_op) - t2_op_ref << "\n";
            std::cout << "  xTAx with t2_op_ref: " << x_.transpose() * t2_op_ref * x_ << "\n";
            */
            assert(xTAx_err < std::sqrt(std::numeric_limits<T>::epsilon()));
          }
#endif
          

          // solve sub-problem t2_op * x = t3_rhs
          {
            int r1 = t3_rhs.r1();
            int m = t3_rhs.n();
            int r2 = t3_rhs.r2();
            Eigen::Matrix<T, Eigen::Dynamic, 1> b_(r1*m*r2);
            for(int i = 0; i < r1; i++)
              for(int j = 0; j < m; j++)
                for(int k = 0; k < r2; k++)
                  b_(i + j*r1 + k*m*r1) = t3_rhs(i,j,k);
            assert( b_.size() == t2_op.r1() );

            const Eigen::Matrix<T, Eigen::Dynamic, 1> x_ = ConstEigenMap(t2_op).colPivHouseholderQr().solve(b_);

            if( useMALS == 0 )
            {
              auto& subTx = TTx.editableSubTensors()[iDim];
              assert( subTx.r1() == r1 );
              assert( subTx.r2() == r2 );
              const auto n = subTx.n();
              assert(r1*n*r2 == x_.size());

              if( iDim+1 == nDim )
              {
                for(int i = 0; i < r1; i++)
                  for(int j = 0; j < n; j++)
                    for(int k = 0; k < r2; k++)
                      subTx(i,j,k) = x_(i + j*r1 + k*n*r1);
              }
              else
              {
                Tensor2<T> t2x(r1*n,r2);
                for(int i = 0; i < r1; i++)
                  for(int j = 0; j < n; j++)
                    for(int k = 0; k < r2; k++)
                      t2x(i+j*r1,k) = x_(i + j*r1 + k*n*r1);

                const auto [Q,B] = internal::normalize_qb(t2x);
                const auto r2_new = Q.cols();
                subTx.resize(r1, n, r2_new);
                for(int i = 0; i < r1; i++)
                  for(int j = 0; j < n; j++)
                    for(int k = 0; k < r2_new; k++)
                      subTx(i,j,k) = Q(i+j*r1,k);

                // first contract t2x(:,*) * subT(*,:,:)
                t2x.resize(r2_new,r2);
                Tensor3<T> subTx_next;
                internal::normalize_contract1(t2x, TTx.subTensors()[iDim+1], subTx_next);
                std::swap(TTx.editableSubTensors()[iDim+1], subTx_next);
              }
            }
          }
        }

        // prepare left/right xTb for the next iteration
        {
          const auto& subTb = symmetricA ? TTb.subTensors()[iDim] : TTAtb.subTensors()[iDim];
          const auto& subTx = TTx.subTensors()[iDim];

          Tensor3<T> t3_tmp;
          // first contraction: t2(*,:) * subTb(*,:,:)
          internal::reverse_dot_contract1(left_xTb.back(), subTb, t3_tmp);

          // second contraction: t3(*,*,:) * subTx(*,*,:)
          Tensor2<T> t2;
          internal::reverse_dot_contract2(t3_tmp, subTx, t2);
          left_xTb.emplace_back(std::move(t2));
        }

        // prepare left/right xTAx for the next iteration
        {
          const auto& TTOp = symmetricA ? TTOpA : TTOpAtA;
          const auto& subTOp = TTOp.tensorTrain().subTensors()[iDim];
          const auto& subTx = TTx.subTensors()[iDim];
          auto& Axk = TTAx.editableSubTensors()[iDim];
          internal::apply_contract(TTOp, iDim, subTOp, subTx, Axk);
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
          Tensor3<T> t3;
          // first contraction: xTAx(*,:) * Axk(*,:,:)
          internal::reverse_dot_contract1(left_xTAx.back(), Axk, t3);
          // now we have
          //   __________
          //   |        |
          //  x_k^T --- t3
          //   |        ||
          //
          Tensor2<T> next_xTAx;
          // second contraction: t3(*,*,:) * subTx(*,*,:)
          internal::reverse_dot_contract2(t3, subTx, next_xTAx);

          left_xTAx.emplace_back(std::move(next_xTAx));

          if( iDim+1 < nDim )
          {
            const auto& subTOp_next = TTOp.tensorTrain().subTensors()[iDim+1];
            const auto& subTx_next = TTx.subTensors()[iDim+1];
            auto& Axk_next = TTAx.editableSubTensors()[iDim+1];
            internal::apply_contract(TTOp, iDim+1, subTOp_next, subTx_next, Axk_next);
          }
        }

#ifndef NDEBUG
        if( symmetricA )
        {
          TensorTrain<T> TTAx_ref(TTOpA.row_dimensions());
          apply(TTOpA, TTx, TTAx_ref);
          const auto errAx = std::abs(axpby(1., TTAx, -1., TTAx_ref));
          assert(errAx < std::sqrt(std::numeric_limits<T>::epsilon()));
        }
        else
        {
          TensorTrain<T> TTAtAx_ref(TTOpAtA.row_dimensions());
          apply(TTOpAtA, TTx, TTAtAx_ref);
          const auto errAx = std::abs(axpby(1., TTAx, -1., TTAtAx_ref));
          assert(errAx < std::sqrt(std::numeric_limits<T>::epsilon()));
        }
#endif
      }
#ifndef NDEBUG
    {
      assert(right_xTb.size() == 1);
      assert(left_xTb.size() == nDim+1);
      assert(left_xTb[nDim].r1() == 1);
      assert(left_xTb[nDim].r2() == 1);
      if( symmetricA )
      {
        const auto dot_ref = dot(TTx, TTb);
        const auto err = std::abs(left_xTb[nDim](0,0) - dot_ref);
        assert(err < std::sqrt(std::numeric_limits<T>::epsilon()));
      }
      else
      {
        const auto dot_ref = dot(TTx, TTAtb);
        const auto err = std::abs(left_xTb[nDim](0,0) - dot_ref);
        assert(err < std::sqrt(std::numeric_limits<T>::epsilon()));
      }
    }
#endif
#ifndef NDEBUG
    {
      assert(right_xTAx.size() == 1);
      assert(left_xTAx.size() == nDim+1);
      assert(left_xTAx[nDim].r1() == 1);
      assert(left_xTAx[nDim].r2() == 1);
      if( symmetricA )
      {
        TensorTrain<T> TTAx_ref(TTOpA.row_dimensions());
        apply(TTOpA, TTx, TTAx_ref);
        const auto errAx = std::abs(axpby(1., TTAx, -1., TTAx_ref));
        assert(errAx < std::sqrt(std::numeric_limits<T>::epsilon()));
      }
      else
      {
        TensorTrain<T> TTAtAx_ref(TTOpAtA.row_dimensions());
        apply(TTOpAtA, TTx, TTAtAx_ref);
        const auto errAx = std::abs(axpby(1., TTAx, -1., TTAtAx_ref));
        assert(errAx < std::sqrt(std::numeric_limits<T>::epsilon()));
      }
      const auto dot_ref = dot(TTx, TTAx);
      const auto err = std::abs(left_xTAx[nDim](0,0) - dot_ref);
      assert(err < std::sqrt(std::numeric_limits<T>::epsilon()));
    }
#endif

      if( symmetricA )
        squaredError = bTb + dot(TTAx,TTAx) - 2*dot(TTAx,TTb);
      else
        squaredError = bTb + dot(TTAx,TTAx) - 2*dot(TTAx,TTAtb);
      residualError = std::sqrt(std::abs(squaredError));
      if( symmetricA )
        std::cout << "Sweep " << iSweep+0.5 << " residual norm: " << residualError << "\n";
      else
        std::cout << "Sweep " << iSweep+0.5 << " residual norm of the normal equations: " << residualError << "\n";
    }


    return residualError;
  }

}


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_HPP
