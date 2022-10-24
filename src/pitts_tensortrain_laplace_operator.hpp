/*! @file pitts_tensortrain_laplace_operator.hpp
* @brief Simple stencils for a discretized Laplace operator applied to a tensor-train
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-11-12
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_LAPLACE_OPERATOR_HPP
#define PITTS_TENSORTRAIN_LAPLACE_OPERATOR_HPP

// includes
//#include <omp.h>
//#include <iostream>
#include <cmath>
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Apply an n-dimensional Laplace operator to a tensor in tensor-train format
  //!
  //! Based on a simple finite-difference discretization for the n-dimensional unit square /f$ [0,1]^n /f$.
  //! This results in the stencil /f$ 1 / \Delta_x^2 (1,-2,1) /f$ in each direction.
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param[inout] TT tensor in tensor-train format, result is normalized
  //! @return          norm of the resulting tensor
  //!
  template<typename T>
  T laplaceOperator(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()))
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // laplace Operator has the form
    //
    //   A x I x I x ... x I
    // + I x A x I x ... x I
    // + ..
    // + I x I x I x ... x A
    // 
    // where I is the identity operator and A=1/(n+1)*tridiag(1,-2,1)
    //

    // stupid implementation
    Tensor3<T> newSubT, oldSubT;
    TensorTrain<T> tmpTT = TT;
    T norm = T(0);
    for(int iDim = 0; iDim < TT.dimensions().size(); iDim++)
    {
      copy(tmpTT.subTensor(iDim), oldSubT);
      const auto r1 = oldSubT.r1();
      const auto r2 = oldSubT.r2();
      const auto n = oldSubT.n();
      newSubT.resize(r1,n,r2);
      // central part
      for(int k = 0; k < n; k++)
        for(int j = 0; j < r2; j++)
          for(int i = 0; i < r1; i++)
            newSubT(i,k,j) = -2 * oldSubT(i,k,j) / (n+1);
      newSubT = tmpTT.setSubTensor(iDim, std::move(newSubT));
      norm = axpby(T(1), tmpTT, norm, TT, rankTolerance);
      
      // left part
      for(int k = n-1; k >= 0; k--)
        for(int j = 0; j < r2; j++)
          for(int i = 0; i < r1; i++)
            newSubT(i,k,j) = (k-1) >= 0 ? oldSubT(i,k-1,j) / (n+1) : T(0);
      newSubT = tmpTT.setSubTensor(iDim, std::move(newSubT));
      norm = axpby(T(1), tmpTT, norm, TT, rankTolerance);

      // right part
      for(int k = 0; k < n; k++)
        for(int j = 0; j < r2; j++)
          for(int i = 0; i < r1; i++)
            newSubT(i,k,j) = (k+1) < n ? oldSubT(i,k+1,j) / (n+1) : T(0);
      newSubT = tmpTT.setSubTensor(iDim, std::move(newSubT));
      norm = axpby(T(1), tmpTT, norm, TT, rankTolerance);
      
      // revert
      oldSubT = tmpTT.setSubTensor(iDim, std::move(oldSubT));
    }

    return norm;
  }

}


#endif // PITTS_TENSORTRAIN_LAPLACE_OPERATOR_HPP
