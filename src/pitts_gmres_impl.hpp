// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_gmres_impl.hpp
* @brief Generic iterative solver for linear systems based on GMRES with templated underlying vector and matrix data type
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
*
**/

// include guard
#ifndef PITTS_GMRES_IMPL_HPP
#define PITTS_GMRES_IMPL_HPP

// includes
#include <cmath>
#include <limits>
#include <cassert>
#include <iostream>
#include <vector>
#include "pitts_gmres.hpp"
#include "pitts_timer.hpp"
#include "pitts_eigen.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! dispatch abs to either std::abs(x) or x.abs() (for compatibility with Eigen Arrays)
    template<typename T>
    auto abs(const T& x)
    {
      constexpr bool has_abs = requires(const T& t){t.abs();};
      if constexpr (has_abs)
        return x.abs();
      else
        return std::abs(x);
    }

    //! dispatch sqrt to either std::sqrt(x) or x.sqrt() (for compatibility with Eigen Arrays)
    template<typename T>
    auto sqrt(const T& x)
    {
      constexpr bool has_sqrt = requires(const T& t){t.sqrt();};
      if constexpr (has_sqrt)
        return x.sqrt();
      else
        return std::sqrt(x);
    }

    //! dispatch any to either (x) or x.any() (for compatibility with boolean Eigen Array expressions)
    template<typename T>
    auto any(const T& x)
    {
      constexpr bool has_any = requires(const T& t){t.any();};
      if constexpr (has_any)
        return x.any();
      else
        return x;
    }

    //! backward substitution (upper triangular solve)
    //!
    //! Eigen internal solve method does not cope well with Eigen Arrays as scalar type...
    //!
    template<typename T>
    auto backwardSolve(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& U, const Eigen::Matrix<T, Eigen::Dynamic, 1>& y)
    {
      assert(U.rows() == U.cols());
      const auto n = U.rows();

      auto x = y;
      for(int i = n-1; i >= 0; i--)
      {
        for(int j = i+1; j < n; j++)
          x(i) -= U(i,j) * x(j);
        x(i) /= U(i,i);
      }
      
      return x;
    }
  }

  // implementation of PITTS::GMRES
  template<typename T, typename LinearOperator, typename Vector>
  std::pair<T,T> GMRES(const LinearOperator& OpA, bool symmetric, const Vector& b, Vector& x, int maxIter, const T& absResTol, const T& relResTol, const std::string_view& outputPrefix, bool verbose)
  {
    using vec = Eigen::Matrix<T,Eigen::Dynamic,1>;
    using mat = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

    const auto timer = PITTS::timing::createScopedTimer();

    std::vector<Vector> v;
    auto r0 = Vector();
    apply(OpA, x, r0);
    const auto nrm_b = norm2(b); // we need some prototype to generate T(-1) (could be an Eigen Array)
    const T zero = nrm_b-nrm_b;  // generates zero with the same length as nrm_b
    const T one = zero + 1; // generates one with the same length as nrm_b
    const T minusOne = zero - 1; // generates minusOne with the same length as nrm_b
    const auto beta = axpy_norm2(minusOne, b, r0);
    if( verbose )
      std::cout << outputPrefix << "Initial residual norm: " << beta << ", rhs norm: " << nrm_b << "\n";
    if( internal::any( beta <= absResTol ) )
      return {beta, one};
    scale(T(1/beta), r0);
    v.emplace_back(std::move(r0));

    vec b_hat = vec::Zero(maxIter+1);
    b_hat(0) = beta;
    mat H = mat::Constant(maxIter+1, maxIter, zero);
    mat R = mat::Constant(maxIter, maxIter, zero);
    vec c = vec::Constant(maxIter, zero);
    vec s = vec::Constant(maxIter, zero);

    // current residual
    T rho = beta;

    for(int i = 0; i < maxIter; i++)
    {
      auto w = Vector();
      apply(OpA, v[i], w);

      // fused axpy and dot/norm2 operations to increase the computational intensity
      H(i,i) = dot(v[i], w);
      const int jmin = symmetric ? std::max(0,i-1) : 0;
      for(int j = i; j > jmin; j--)
        H(j-1,i) = axpy_dot(T(-H(j,i)), v[j], w, v[j-1]);
      H(i+1,i) = axpy_norm2(T(-H(jmin,i)), v[jmin], w);
      scale(T(1/H(i+1,i)), w);
      v.emplace_back(std::move(w));

      // least squares solve using Givens rotations
      R(0,i) = H(0,i);
      for(int j = 1; j <= i; j++)
      {
        const T gamma = c(j-1)*R(j-1,i) + s(j-1)*H(j,i);
        R(j,i) = -s(j-1)*R(j-1,i) + c(j-1)*H(j,i);
        R(j-1,i) = gamma;
      }
      const T delta = internal::sqrt(R(i,i)*R(i,i) + H(i+1,i)*H(i+1,i));
      c(i) = R(i,i) / delta;
      s(i) = H(i+1,i) / delta;
      R(i,i) = c(i)*R(i,i) + s(i)*H(i+1,i);
      b_hat(i+1) = -s(i)*b_hat(i);
      b_hat(i) = c(i)*b_hat(i);
      rho = internal::abs(b_hat(i+1));
      if( verbose )
        std::cout << outputPrefix << "GMRES iteration " << i+1 << " residual: " << rho << "\n";

      // check convergence
      if( internal::any( rho <= absResTol || rho/beta <= relResTol ) )
        break;
    }

    // calculate solution
    const int m = v.size() - 1;
    const mat Ri = R.topLeftCorner(m, m);
    const vec bi = b_hat.topRows(m);
    const vec y = internal::backwardSolve(Ri, bi);//Ri.template triangularView<Eigen::Upper>().solve(bi);
    for(int j = 0; j < m; j++)
      axpy(T(-y(j)), v[j], x);

    return {rho, rho/beta};
  }

}


#endif // PITTS_GMRES_IMPL_HPP
