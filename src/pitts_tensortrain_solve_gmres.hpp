/*! @file pitts_tensortrain_solve_gmres.hpp
* @brief TT-GMRES algorithm, iterative solver for linear systems in tensor-train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-07-04
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SOLVE_GMRES_HPP
#define PITTS_TENSORTRAIN_SOLVE_GMRES_HPP

// includes
#include <iostream>
#include <cassert>
#include <vector>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_gram_schmidt.hpp"
#include "pitts_gmres.hpp"
#include "pitts_timer.hpp"
#pragma GCC push_options
#pragma GCC optimize("no-unsafe-math-optimizations")
#include <Eigen/Dense>
#pragma GCC pop_options

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
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
  }

  //! TT-GMRES: iterative solver for linear systems in tensor-train format
  //!
  //! Given A, b, calculates x with Ax=b approximately up to a given tolerance.
  //!
  //! @tparam T             data type (double, float, complex)
  //!
  //! @param TTOpA              tensor-train operator A
  //! @param TTb                right-hand side tensor-train b
  //! @param TTx                initial guess on input, overwritten with the (approximate) result on output
  //! @param maxIter            maximal number of iterations
  //! @param absResTol          absolute residual tolerance: the iteration aborts if the absolute residual norm is smaller than absResTol
  //! @param relResTol          relative residual tolerance: the iteration aborts if the relative residual norm is smaller than relResTol
  //! @param maxRank            maximal allowed TT-rank, enforced even if this violates the residualTolerance
  //! @param adaptiveTolerance  use an adaptive tolerance for the tensor-train arithmetic in the iteration
  //! @param outputPrefix       string to prefix all output about the convergence history
  //! @param verbose            set to true, to print the residual norm in each iteration to std::cout
  //! @return                   residual norm of the result (||Ax - b||)
  //!
  template <typename T>
  auto solveGMRES(const TensorTrainOperator<T> &TTOpA, const TensorTrain<T> &TTb, TensorTrain<T> &TTx,
                  int maxIter, T absResTol, T relResTol,
                  int maxRank = std::numeric_limits<int>::max(), bool adaptiveTolerance = true,
                  const std::string &outputPrefix = "", bool verbose = false)
  {
    using vec = Eigen::Matrix<T,Eigen::Dynamic,1>;
    using mat = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // RHS norm (just for information, might be useful to adjust tolerances...)
    const T nrm_b = norm2(TTb);
    const T r0_rankTol = relResTol / maxIter * T(0.1);

    std::vector<TensorTrain<T>> V;
    // calculate the initial residual
    V.emplace_back(TensorTrain<T>{TTb.dimensions()});
    apply(TTOpA, TTx, V[0]);
    const T beta = axpby(T(-1), TTb, T(1), V[0], r0_rankTol, maxRank);
    if( verbose )
      std::cout << outputPrefix << "Initial residual norm: " << beta << " (abs), " << beta / beta << " (rel), rhs norm: " << nrm_b << ", ranks: " << internal::to_string(V[0].getTTranks()) << "\n";
    
    if( beta <= absResTol )
      return beta;

    // relative residual tolerance used below also for calculating the required TT rankTolerance
    const T residualTolerance = std::max(relResTol, absResTol/beta);

    vec b_hat = vec::Zero(maxIter+1);
    b_hat(0) = beta;
    mat H = mat::Zero(maxIter+1, maxIter);
    mat R = mat::Zero(maxIter, maxIter);
    vec c = vec::Zero(maxIter);
    vec s = vec::Zero(maxIter);

    // current residual
    T rho = beta;

    for(int i = 0; i < maxIter; i++)
    {
      TensorTrain<T> w(TTb.dimensions());
      apply(TTOpA, V[i], w);

      T rankTolerance = residualTolerance / maxIter / T(1.1);
      if( adaptiveTolerance )
        rankTolerance *= beta / rho;

      H.col(i).segment(0, i+2) = gramSchmidt(V, w, rankTolerance, maxRank, outputPrefix + "  gramSchmidt: ", verbose);

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
      b_hat(i+1) = -s(i)*b_hat(i);
      b_hat(i) = c(i)*b_hat(i);
      rho = std::abs(b_hat(i+1));
      if( verbose )
        std::cout << outputPrefix << "TT-GMRES iteration " << i+1 << " residual norm: " << rho << " (abs), " << rho / beta << " (rel), ranks: " << internal::to_string(V[0].getTTranks()) << "\n";

      // check convergence
      if( rho/beta <= residualTolerance )
        break;
    }

    // calculate solution
    const int m = V.size() - 1;
    const mat Ri = R.topLeftCorner(m, m);
    const vec bi = b_hat.topRows(m);
    const vec y = internal::backwardSolve(Ri, bi);
    T nrm_x = 1;
    for(int j = 0; j < m; j++)
    {
      const T rankTolerance = residualTolerance / m / T(1.1) * std::min(T(1), beta/nrm_b);
      nrm_x = axpby(T(-y(j)), V[j], nrm_x, TTx, rankTolerance, maxRank);
    }
    internal::t3_scale(nrm_x, TTx.editableSubTensors()[0]);

    /*
    // check accuracy of the result (for debugging problems with the rank tolerances...)
    apply(TTOpA, TTx, V[0]);
    T rho_ref = axpby(T(-1), TTb, nrm_x, V[0], T(0));
    if( verbose )
      std::cout << outputPrefix << "Real residual norm: " << rho << " (abs), " << rho / beta << " (rel), x norm: " << nrm_x << ", ranks: " << internal::to_string(TTx.getTTranks()) << "\n";
    */

    return rho;
  }

}


#endif // PITTS_TENSORTRAIN_SOLVE_GMRES_HPP
