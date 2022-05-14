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

    //! axpy operation for two Tensor3 (with matching dimensions)
    template<typename T>
    void t3_axpy(T alpha, const Tensor3<T>& x, Tensor3<T>& y)
    {
      const auto r1 = x.r1();
      const auto n = x.n();
      const auto r2 = x.r2();
      assert(x.r1() == y.r1());
      assert(x.n() == y.n());
      assert(x.r2() == y.r2());

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "n", "r2"},{r1, n, r2}}, // arguments
        {{r1*n*r2*kernel_info::FMA<T>()}, // flops
         {(r1*n*r2)*kernel_info::Load<T>() + (r1*n*r2)*kernel_info::Update<T>()}} // data transfers
        );

      for(int i = 0; i < r1; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2; k++)
            y(i,j,k) += alpha * x(i,j,k);
    }

    template<typename T>
    void als_apply_op_contract1(const Tensor2<T>& lOp, const int rA1, const Tensor3<T>& t3x, Tensor3<T>& t3tmp)
    {
      const auto r1 = lOp.r1() / rA1;
      const auto r = lOp.r2();
      assert(lOp.r2() == t3x.r1());
      const auto n = t3x.n();
      const auto r2 = t3x.r2();

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "r", "n", "r2", "rA1"},{r1, r, n, r2, rA1}}, // arguments
        {{r1*n*rA1*r2*r*kernel_info::FMA<T>()}, // flops
         {(r1*rA1*r + r*n*r2)*kernel_info::Load<T>() + (r1*rA1*n*r2)*kernel_info::Store<T>()}} // data transfers
        );

      t3tmp.resize(r1, rA1*n, r2);
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < rA1; j++)
          for(int k = 0; k < n; k++)
            for(int l = 0; l < r2; l++)
            {
              T tmp = T(0);
              for(int ii = 0; ii < r; ii++)
                tmp += lOp(ii+j*r1,i) * t3x(ii,k,l);
              t3tmp(i,j+k*rA1,l) = tmp;
            }
    }

    template<typename T>
    void als_apply_op_contract2(const Tensor3<T>& Ak, const int m, const Tensor3<T>& t3a, Tensor3<T>& t3b)
    {
      const auto r1 = t3a.r1();
      const auto r2 = t3a.r2();
      const auto rA1 = t3a.n() / m;
      const auto n = Ak.n() / m;
      const auto rA2 = Ak.r2();
      assert(rA1 == Ak.r1());

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "r2", "rA1", "n", "m", "rA2"},{r1, r2, rA1, n, m, rA2}}, // arguments
        {{r1*n*rA2*r2*rA1*m*kernel_info::FMA<T>()}, // flops
         {(rA1*n*m*rA2 + r1*rA1*m*r2)*kernel_info::Load<T>() + (r1*n*rA2*r2)*kernel_info::Store<T>()}} // data transfers
        );

      t3b.resize(r1, n, rA2*r2);
#pragma omp parallel for collapse(4) schedule(static)
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < rA2; k++)
            for(int l = 0; l < r2; l++)
            {
              T tmp = T(0);
              for(int ii = 0; ii < rA1; ii++)
                for(int jj = 0; jj < m; jj++)
                  tmp += Ak(ii,j+jj*n,k) * t3a(i,ii+jj*rA1,l);
              t3b(i,j,k+l*rA2) = tmp;
            }
    }

    template<typename T>
    void als_apply_op_contract3(const Tensor3<T>& t3tmp, const Tensor2<T>& rOp, const int rA2, Tensor3<T>& t3y)
    {
      const auto r1 = t3tmp.r1();
      const auto n = t3tmp.n();
      const auto r2 = rOp.r1();
      const auto r = rOp.r2() / rA2;
      assert(r*rA2 == t3tmp.r2());

      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1", "n", "r2", "r", "rA2"},{r1, n, r2, r, rA2}}, // arguments
        {{r1*n*r2*rA2*r*kernel_info::FMA<T>()}, // flops
         {(r1*n*r*rA2 + r2*r*rA2)*kernel_info::Load<T>() + (r1*n*r2)*kernel_info::Store<T>()}} // data transfers
        );

      t3y.resize(r1,n,r2);
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2; k++)
          {
            T tmp = T(0);
            for(int ii = 0; ii < rA2; ii++)
              for(int jj = 0; jj < r; jj++)
                tmp += t3tmp(i,j,ii+jj*rA2) * rOp(k,jj+ii*r);
            t3y(i,j,k) = tmp;
          }
    }

    //! calculate the result of multiplying two rank-3 operator tensors (contraction of third and first dimension, reordering of middle dimension)
    //!
    //! The resulting tensor is t3c with
    //!   t3c_(i,(k,l),j) = sum_l t3a_(i,(k1,l1),l) * t3b_(l,(k2,l2),j)
    //! with a tensor product of the second dimensions.
    //!
    //! @tparam T  underlying data type (double, complex, ...)
    //!
    //! @param t3a    first rank-3 tensor
    //! @param t3b    second rank-3 tensor
    //! @param na     first middle dimensions of t3a
    //! @param nb     first middle dimensions of t3b
    //! @return       resulting t3c (see formula above)
    //!
    template<typename T>
    auto combine_op(const Tensor3<T>& t3a, const Tensor3<T>& t3b, int na, int nb)
    {
      if( t3a.r2() != t3b.r1() )
        throw std::invalid_argument("Dimension mismatch!");

      // gather performance data
      const auto r1 = t3a.r1();
      assert(t3a.n() % na == 0);
      const auto ma = t3a.n() / na;
      const auto r = t3a.r2();
      assert(t3b.n() % nb == 0);
      const auto mb = t3b.n() / nb;
      const auto r2 = t3b.r2();

      const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
          {{"r1", "na", "ma", "r", "nb", "mb", "r2"}, {r1, na, ma, r, nb, mb, r2}},     // arguments
          {{r1*r2*na*ma*nb*mb*r*kernel_info::FMA<T>()},    // flops
          {(r1*na*ma*r + r*nb*mb*r2)*kernel_info::Load<T>() + r1*na*ma*nb*mb*r2*kernel_info::Store<T>()}}    // data transfers
          );

      Tensor3<T> t3c(r1,na*ma*nb*mb,r2);
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < r2; j++)
          for(int k1 = 0; k1 < na; k1++)
            for(int l1 = 0; l1 < ma; l1++)
              for(int k2 = 0; k2 < nb; k2++)
                for(int l2 = 0; l2 < mb; l2++)
                {
                  T tmp = T(0);
                  for(int ii = 0; ii < r; ii++)
                    tmp += t3a(i,k1+na*l1,ii) * t3b(ii,k2+nb*l2,j);
                  t3c(i,(k1+na*k2)+(l1+ma*l2)*na*nb,j) = tmp;
                }
      return t3c;
    }


    //! dedicated helper functions for solveMALS
    namespace solve_mals
    {
      //! helper function for converting an array to a string
      template<typename T>
      std::string to_string(const std::vector<T>& v)
      {
        std::string result = "[";
        for(int i = 0; i < v.size(); i++)
        {
          if( i > 0 )
            result += ", ";
          result += std::to_string(v[i]);
        }
        result += "]";
        return result;
      }

      //! Tensor3 as vector
      template<typename T>
      auto flatten(const Tensor3<T>& t3)
      {
        Eigen::Matrix<T, Eigen::Dynamic, 1> v(t3.r1()*t3.n()*t3.r2());
        for(int i = 0; i < t3.r1(); i++)
          for(int j = 0; j < t3.n(); j++)
            for(int k = 0; k < t3.r2(); k++)
              v(i + j*t3.r1() + k*t3.n()*t3.r1()) = t3(i,j,k);
        return v;
      };

      //! vector as Tensor3 (with known dimensions)
      template<typename T>
      void unflatten(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v, Tensor3<T>& t3)
      {
        assert(v.size() == t3.r1()*t3.n()*t3.r2());
        for(int i = 0; i < t3.r1(); i++)
          for(int j = 0; j < t3.n(); j++)
            for(int k = 0; k < t3.r2(); k++)
              t3(i,j,k) = v(i + j*t3.r1() + k*t3.n()*t3.r1());
      }

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

      //! calculate the local RHS vector for ALS
      //! --- left_xTb
      //!        |
      //!   --  b_k
      //!        |
      //! --- right_xTb
      template<typename T>
      Tensor3<T> calculate_local_rhs(const Tensor2<T>& left_xTb, const Tensor3<T>& subTb, const Tensor2<T>& right_xTb)
      {

        // first contract: subTb(:,:,*) * t2(:,*)
        Tensor3<T> t3_tmp;
        internal::dot_contract1(subTb, right_xTb, t3_tmp);

        // then contract: t2(*,:) * t3_tmp(*,:,:)
        Tensor3<T> t3_rhs;
        internal::reverse_dot_contract1(left_xTb, t3_tmp, t3_rhs);
        return t3_rhs;
      }

      //! apply sub-problem operator
      /*!
       *! --- left_xTAx ---       \
       *!        |                 \
       *!   --  A_k --        *  -- t3_x
       *!        |                 /
       *! --- right_xTAx ---      /
      */
      template<typename T>
      void apply_local_op(const Tensor2<T>& left_xTAx, const Tensor3<T>& Ak, int nAk, const Tensor2<T>& right_xTAx, const Tensor3<T>& t3_x, Tensor3<T>& t3_y)
      {
        // first contract left_xTAx and t3_x
        internal::als_apply_op_contract1(left_xTAx, Ak.r1(), t3_x, t3_y);

        // now contract t3_tmp and Ak
        Tensor3<T> t3_tmp;
        internal::als_apply_op_contract2(Ak, nAk, t3_y, t3_tmp);

        // finally contract result with right_xTAx
        internal::als_apply_op_contract3(t3_tmp, right_xTAx, Ak.r2(), t3_y);
      }

      //! GMRES implementation for the local problem (without restart)
      //!
      //! We could use MINRES but GMRES is slightly more robust and the additional operations (dot, axpy) are currently not influencing the total runtime significantly.
      //!
      template<typename T>
      void solve_local_GMRES(const Tensor2<T>& localOp_left_xTAx, const Tensor3<T>& localOp_Ak, int nAk, const Tensor2<T>& localOp_right_xTAx,
                             const Tensor3<T>& t3_rhs, Tensor3<T>& t3_x, const int maxIter, const T absResTol, const T relResTol)
      {
        using vec = Eigen::Matrix<T,Eigen::Dynamic,1>;
        using mat = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

        const auto apply_A = [&localOp_left_xTAx, &localOp_Ak, &nAk, &localOp_right_xTAx](const Tensor3<T>& x)
        {
          Tensor3<T> Ax;
          apply_local_op(localOp_left_xTAx, localOp_Ak, nAk, localOp_right_xTAx, x, Ax);
          return Ax;
        };

        auto t3_tmp = apply_A(t3_x);
        t3_axpy(T(-1), t3_rhs, t3_tmp);
        const T beta = t3_nrm(t3_tmp);
        std::cout << " local problem: initial residual: " << beta << "\n";
        if( beta <= absResTol )
          return;
        t3_scale(1/beta, t3_tmp);

        vec b_hat = vec::Unit(maxIter+1, 0) * beta;
        mat H = mat::Zero(maxIter+1,maxIter);
        mat R = mat::Zero(maxIter,maxIter);
        vec c = vec::Zero(maxIter);
        vec s = vec::Zero(maxIter);
        
        std::vector<Tensor3<T>> v;
        v.emplace_back(std::move(t3_tmp));


        for(int i = 0; i < maxIter; i++)
        {
          auto w = apply_A(v[i]);
          for(int j = 0; j <= i; j++)
          {
            H(j,i) = t3_dot(v[j],w);
            t3_axpy(-H(j,i),v[j],w);
          }
          H(i+1,i) = t3_nrm(w);
          t3_scale(1/H(i+1,i), w);
          v.emplace_back(std::move(w));
          //std::cout << " local problem: H:\n" << H.topLeftCorner(i+2,i+1) << "\n";

          // least squares solve using Givens rotations
          R(0,i) = H(0,i);
          for(int j = 1; j <= i; j++)
          {
            const T gamma = c(j-1)*R(j-1,i) + s(j-1)*H(j,i);
            R(j,i) = -s(j-1)*R(j-1,i) + c(j-1)*H(j,i);
            R(j-1,i) = gamma;
          }
          const T delta = std::sqrt(R(i,i)*R(i,i) + H(i+1,i)*H(i+1,i));
          c(i) = R(i,i) / delta;
          s(i) = H(i+1,i) / delta;
          R(i,i) = c(i)*R(i,i) + s(i)*H(i+1,i);
          //std::cout << " local problem: c: " << c.topRows(i+1).transpose() << "\n";
          //std::cout << " local problem: s: " << s.topRows(i+1).transpose() << "\n";
          //std::cout << " local problem: R:\n" << R.topLeftCorner(i+1,i+1) << "\n";
          b_hat(i+1) = -s(i)*b_hat(i);
          b_hat(i) = c(i)*b_hat(i);
          //std::cout << " local problem: b_hat: " << b_hat.transpose() << "\n";
          const T rho = std::abs(b_hat(i+1));
          std::cout << " local problem: GMRES iter " << i+1 << " residual: " << rho << "\n";
          if( rho <= absResTol || rho/beta <= relResTol )
          {
            const auto Ri = R.topLeftCorner(i+1,i+1);
            const auto bi = b_hat.topRows(i+1);
            const vec y = Ri.template triangularView<Eigen::Upper>().solve(b_hat.topRows(i+1));
            //std::cout << " local problem: y: " << y.transpose() << "\n";
            for(int j = 0; j <= i; j++)
              t3_axpy(-y(j),v[j],t3_x);
            break;
          }
        }
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
  //! @param MALS               set to false to use ALS, true for MALS
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
              bool MALS = true)
  {
    using mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using namespace internal::solve_mals;

    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // check that dimensions match
    if( TTb.dimensions() != TTOpA.row_dimensions() )
      throw std::invalid_argument("TensorTrain solveMALS dimension mismatch!");
    if( TTx.dimensions() != TTOpA.column_dimensions() )
      throw std::invalid_argument("TensorTrain solveMALS dimension mismatch!");

    const int nDim = TTx.subTensors().size();
    const bool useMALS = nDim > 1 && MALS;

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
    TensorTrain<T> TTAx(effTTOpA.row_dimensions());
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
    // ||Ax - b||_2^2 = <Ax - b, Ax - b> = <Ax,Ax> - 2<Ax, b> + <b,b>
    T squaredError = bTb + dot(TTAx,TTAx) - 2*dot(TTAx,effTTb);
    auto residualError = std::sqrt(std::abs(squaredError));
    std::cout << "Initial residual norm: " << residualError << " (abs), " << residualError / sqrt_bTb << " (rel), ranks: " << to_string(TTx.getTTranks()) << "\n";

    // now everything is prepared, perform the sweeps
    for(int iSweep = 0; iSweep < nSweeps; iSweep++)
    {
      if( residualError / sqrt_bTb < residualTolerance )
        break;

      if( useMALS )
      {
        right_xTb.pop_back();
        right_xTAx.pop_back();
      }

      // sweep left to right
      for(int iDim = 0; iDim < nDim; iDim++)
      {
        if( iDim + useMALS < nDim )
        {
          right_xTb.pop_back();
          right_xTAx.pop_back();

          Tensor3<T> t3x, t3b;
          Tensor3<T> t3A;
          int nA;
          if( !useMALS )
          {
            copy(effTTb.subTensors()[iDim], t3b);
            copy(TTx.subTensors()[iDim], t3x);
            copy(effTTOpA.tensorTrain().subTensors()[iDim], t3A);
            nA = effTTOpA.row_dimensions()[iDim];
          }
          else // useMALS
          {
            copy(combine(effTTb.subTensors()[iDim], effTTb.subTensors()[iDim+1]), t3b);
            copy(combine(TTx.subTensors()[iDim], TTx.subTensors()[iDim+1]), t3x);
            const int n1 = effTTOpA.row_dimensions()[iDim];
            const int n2 = effTTOpA.row_dimensions()[iDim+1];
            copy(internal::combine_op(effTTOpA.tensorTrain().subTensors()[iDim], effTTOpA.tensorTrain().subTensors()[iDim+1], n1, n2), t3A);
            nA = n1*n2;
          }

          // calculate sub-problem right-hand side
          const Tensor3<T> t3_rhs = calculate_local_rhs(left_xTb.back(), t3b, right_xTb.back());
          assert( std::abs( internal::t3_dot(t3x, t3_rhs) - dot(TTx, effTTb) ) < sqrt_eps );

#ifndef NDEBUG
          {
            Tensor3<T> t3Ax;
            apply_local_op(left_xTAx.back(), t3A, nA, right_xTAx.back(), t3x, t3Ax);
            //assert( std::abs( internal::t3_dot(t3x, t3Ax) - dot(TTx,TTAx) ) < sqrt_eps );
          }
#endif
          // absolute tolerance is not invariant wrt. #dimensions
          solve_local_GMRES(left_xTAx.back(), t3A, nA, right_xTAx.back(), t3_rhs, t3x, 50, T(0), T(1.e-4));
          const vec x = flatten(t3x);

          if( !useMALS )
          {
            auto& subTx = TTx.editableSubTensors()[iDim];
            const auto r1 = subTx.r1();
            const auto n = subTx.n();
            const auto r2 = subTx.r2();

            if( iDim+1 == nDim )
            {
              unflatten(x, subTx);
            }
            else
            {
              Tensor2<T> t2x(r1*n,r2);
              EigenMap(t2x) = Eigen::Map<const mat>(x.data(), r1*n, r2);

              const auto [Q,B] = internal::normalize_qb(t2x, residualTolerance/nDim, maxRank);
              const auto r2_new = Q.cols();
              subTx.resize(r1, n, r2_new);
              unflatten<T>(Eigen::Map<const vec>(Q.data(), r1*n*r2_new), subTx);

              // now contract B(:,*) * subT(*,:,:)
              t2x.resize(r2_new,r2);
              EigenMap(t2x) = B;
              Tensor3<T> subTx_next;
              internal::normalize_contract1(t2x, TTx.subTensors()[iDim+1], subTx_next);
              std::swap(TTx.editableSubTensors()[iDim+1], subTx_next);
            }
          }
          else // useMALS
          {
            unflatten(x, t3x);
            const int n1 = TTx.dimensions()[iDim];
            const int n2 = TTx.dimensions()[iDim+1];
            auto [xk,xk_next] = split(t3x, n1, n2, true, residualTolerance/nDim, maxRank);
            std::swap(TTx.editableSubTensors()[iDim], xk);
            std::swap(TTx.editableSubTensors()[iDim+1], xk_next);
          }

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
      squaredError = bTb + dot(TTAx,TTAx) - 2*dot(TTAx,effTTb);
      residualError = std::sqrt(std::abs(squaredError));
      std::cout << "Sweep " << iSweep+0.5 << " residual norm: " << residualError << " (abs), " << residualError / sqrt_bTb << " (rel), ranks: " << to_string(TTx.getTTranks()) << "\n";
      if( residualError / sqrt_bTb < residualTolerance )
        break;

      if( useMALS )
      {
        left_xTb.pop_back();
        left_xTAx.pop_back();
      }

      // sweep right to left
      for(int iDim = nDim-1; iDim >= 0; iDim--)
      {
        if( iDim - useMALS >= 0 )
        {
          left_xTb.pop_back();
          left_xTAx.pop_back();

          Tensor3<T> t3x, t3b;
          Tensor3<T> t3A;
          int nA;
          if( !useMALS )
          {
            copy(effTTb.subTensors()[iDim], t3b);
            copy(TTx.subTensors()[iDim], t3x);
            copy(effTTOpA.tensorTrain().subTensors()[iDim], t3A);
            nA = effTTOpA.row_dimensions()[iDim];
          }
          else // useMALS
          {
            copy(combine(effTTb.subTensors()[iDim-1], effTTb.subTensors()[iDim]), t3b);
            copy(combine(TTx.subTensors()[iDim-1], TTx.subTensors()[iDim]), t3x);
            const int n1 = effTTOpA.row_dimensions()[iDim-1];
            const int n2 = effTTOpA.row_dimensions()[iDim];
            copy(internal::combine_op(effTTOpA.tensorTrain().subTensors()[iDim-1], effTTOpA.tensorTrain().subTensors()[iDim], n1, n2), t3A);
            nA = n1*n2;
          }

          // calculate sub-problem right-hand side
          const Tensor3<T> t3_rhs = calculate_local_rhs(left_xTb.back(), t3b, right_xTb.back());
          assert( std::abs( internal::t3_dot(t3x, t3_rhs) - dot(TTx, effTTb) ) < sqrt_eps );

          // check sub-problem operator
#ifndef NDEBUG
          {
            Tensor3<T> t3Ax;
            apply_local_op(left_xTAx.back(), t3A, nA, right_xTAx.back(), t3x, t3Ax);
            std::cout << "error: " << std::abs( internal::t3_dot(t3x, t3Ax) - dot(TTx,TTAx) ) << ", ref: " << dot(TTx,TTAx) << "\n";
            assert( std::abs( internal::t3_dot(t3x, t3Ax) - dot(TTx,TTAx) ) < sqrt_eps );
          }
#endif
          // absolute tolerance is not invariant wrt. #dimensions
          solve_local_GMRES(left_xTAx.back(), t3A, nA, right_xTAx.back(), t3_rhs, t3x, 50, T(0), T(1.e-4));
          const vec x = flatten(t3x);


          if( !useMALS )
          {
            auto& subTx = TTx.editableSubTensors()[iDim];
            const auto r1 = subTx.r1();
            const auto n = subTx.n();
            const auto r2 = subTx.r2();

            if( iDim == 0 )
            {
              unflatten(x, subTx);
            }
            else
            {
              Tensor2<T> t2x(r2*n,r1);
              EigenMap(t2x) = Eigen::Map<const mat>(x.data(), r1, n*r2).transpose();

              const auto [Q,B] = internal::normalize_qb(t2x, residualTolerance/nDim, maxRank);
              const auto r1_new = Q.cols();
              subTx.resize(r1_new, n, r2);
              mat Qt = Q.transpose();
              unflatten<T>(Eigen::Map<const vec>(Qt.data(), r1_new*n*r2), subTx);

              // first contract subT(:,:,*) * B(*,:)
              // now contract: subT(:,:,*) * B^T(:,*)
              t2x.resize(r1_new,r1);
              EigenMap(t2x) = B.transpose();
              Tensor3<T> subTx_prev;
              internal::dot_contract1(TTx.subTensors()[iDim-1], t2x, subTx_prev);
              std::swap(TTx.editableSubTensors()[iDim-1], subTx_prev);
            }
          }
          else // useMALS
          {
            unflatten(x, t3x);
            const int n1 = TTx.dimensions()[iDim-1];
            const int n2 = TTx.dimensions()[iDim];
            auto [xk_prev,xk] = split(t3x, n1, n2, false, residualTolerance/nDim, maxRank);
            std::swap(TTx.editableSubTensors()[iDim-1], xk_prev);
            std::swap(TTx.editableSubTensors()[iDim], xk);
          }
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
      squaredError = bTb + dot(TTAx,TTAx) - 2*dot(TTAx,effTTb);
      residualError = std::sqrt(std::abs(squaredError));
      std::cout << "Sweep " << iSweep+1 << " residual norm: " << residualError << " (abs), " << residualError / sqrt_bTb << " (rel), ranks: " << to_string(TTx.getTTranks()) << "\n";
    }


    return residualError;
  }

}


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_HPP