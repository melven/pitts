/*! @file pitts_tensortrain_solve_gmres_impl.hpp
* @brief TT-GMRES algorithm, iterative solver for linear systems in tensor-train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-07-04
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SOLVE_GMRES_IMPL_HPP
#define PITTS_TENSORTRAIN_SOLVE_GMRES_IMPL_HPP

// includes
#include <string>
#include <iostream>
#include <cassert>
#include <vector>
#include "pitts_tensortrain_solve_gmres.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_gram_schmidt.hpp"
#include "pitts_timer.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain_normalize.hpp"

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

  // implement TT GMRES solver
  template <typename T>
  std::pair<T,T> solveGMRES(const TensorTrainOperator<T> &TTOpA, const TensorTrain<T> &TTb, TensorTrain<T> &TTx,
                            int maxIter, T absResTol, T relResTol, T estimatedCond,
                            int maxRank, bool adaptiveTolerance, bool symmetric,
                            const std::string_view &outputPrefix, bool verbose)
  {
    using vec = Eigen::Matrix<T,Eigen::Dynamic,1>;
    using mat = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    std::string outputPrefix_gramSchmidt = std::string(outputPrefix) + "  gramSchmidt: ";

    // RHS norm (just for information, might be useful to adjust tolerances...)
    const T nrm_b = norm2(TTb);
    const T r0_rankTol = relResTol / maxIter * T(0.1) / estimatedCond;
    const T V0_rankTol = std::numeric_limits<T>::epsilon() * 100;

    std::vector<TensorTrain<T>> V;
    // calculate the initial residual
    V.emplace_back(TensorTrain<T>{TTb.dimensions()});
    apply(TTOpA, TTx, V[0]);
    T tmp = normalize(V[0], V0_rankTol, maxRank);
    const T beta = axpby(T(-1), TTb, tmp, V[0], r0_rankTol, maxRank);
    if( verbose )
      std::cout << outputPrefix << "Initial residual norm: " << beta << " (abs), " << beta / beta << " (rel), rhs norm: " << nrm_b << ", ranks: " << internal::to_string(V[0].getTTranks()) << "\n";
    
    if( beta <= absResTol )
      return {beta, T(1)};

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

      // according to Simoncini2003 (Inexact Krylov methods), this also needs a constant that depends on cond(H) or cond(A)...
      T rankTolerance = residualTolerance / maxIter / T(2.0) / estimatedCond;
      if( adaptiveTolerance )
        rankTolerance *= beta / rho;

      H.col(i).segment(0, i+2) = gramSchmidt(V, w, rankTolerance, maxRank, symmetric, outputPrefix_gramSchmidt, verbose); // for standard MGS: , 1, false, true, false);
      // H(i+1,i) = normalize(w, rankTolerance, maxRank);
      // for(int j = 0; j <= i; j++)
      // {
      //   H(j,i) = H(i+1,i) * dot(V[j], w);
      //   H(i+1,i) = axpby(-H(j,i), V[j], H(i+1,i), w, rankTolerance, maxRank);
      // }
      // V.emplace_back(std::move(w));

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
        std::cout << outputPrefix << "TT-GMRES iteration " << i+1 << " residual norm: " << rho << " (abs), " << rho / beta << " (rel), ranks: " << internal::to_string(V[i+1].getTTranks()) << "\n";

      // check convergence
      // (incorporate possible difference between exact and inexact residual norm)
      if( rho/beta <= residualTolerance / T(2.0) )
        break;
    }

    // calculate solution
    {
      // backward solve
      const int m = V.size() - 1;
      const mat Ri = R.topLeftCorner(m, m);
      const vec bi = b_hat.topRows(m);
      const vec y = Ri.template triangularView<Eigen::Upper>().solve(bi);

      // first add up the delta x (better accuracy as it is probably much smaller than the old x)
      auto& TTdelta_x = V[m-1];
      T nrm_delta_x = T(-y(m-1));
      const T rankTol_x = residualTolerance * std::min(T(1), beta/nrm_b) / estimatedCond;
      for(int j = m-2; j >= 0; j--)
        nrm_delta_x = axpby(T(-y(j)), V[j], nrm_delta_x, TTdelta_x, rankTol_x / m / T(2.0), maxRank);

      // it would be nice to adjust this tolerance here for adding the deltaX to X - but all my variants just made it worse when called from MALS
      const T nrm_x = axpby(nrm_delta_x, TTdelta_x, T(1), TTx, rankTol_x, maxRank);
      TTx.editSubTensor(0, [nrm_x](Tensor3<T>& subT){internal::t3_scale(nrm_x, subT);});
    }

    return {rho, rho/beta};
  }

}


#endif // PITTS_TENSORTRAIN_SOLVE_GMRES_IMPL_HPP
