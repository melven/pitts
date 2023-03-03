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
#include <cmath>
#include <limits>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_operator.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Different variants for defining the sub-problem in MALS-like algorithms for solving linear systems in TT format
  //!
  //! The sub-problem in each step is constructed using an orthogonal projection of the form W^T A V x = W^T b .
  //! This defines different choices for W.
  //!
  enum class MALS_projection
  {
    //! standard choice
    //!
    //! uses W = V, minimizes an energy functional for symmetric positive definite operators.
    //! Might still work for slightly non-symmetric operators.
    //!
    RitzGalerkin = 0,

    //! normal equations
    //!
    //! uses W = A V resulting in the Ritz-Galerkin approach for the normal equations: V^T A^TA V x = V^T A^T b.
    //! Suitable for non-symmetric operators but squares the condition number and doubles the TT ranks in the calculation.
    //! Can be interpreted as a Petrov-Galerkin approach with W = A V.
    //!
    NormalEquations,

    //! non-symmetric approach
    //!
    //! uses an orthogonal W that approximates AV, so A V = W B + E with W^T W = I and W^T E = 0.
    //! Slightly more work but avoids squaring the condition number and the TT ranks in the calculation.
    //! Might suffer from break-downs if the sub-problem operator B = W^T A V is not invertible.
    //!
    PetrovGalerkin
  };


  //! Solve a linear system using the MALS algorithm
  //!
  //! Approximate x with Ax = b
  //!
  //! @tparam T             data type (double, float, complex)
  //!
  //! @param TTOpA              tensor-train operator A
  //! @param symmetric          is the operator symmetric?
  //! @param projection         defines different variants for defining the local sub-problem in the MALS algorithm, for symmetric problems choose RitzGalerkin
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
              bool symmetric,
              const MALS_projection projection,
              const TensorTrain<T>& TTb,
              TensorTrain<T>& TTx,
              int nSweeps,
              T residualTolerance = std::sqrt(std::numeric_limits<T>::epsilon()),
              int maxRank = std::numeric_limits<int>::max(),
              int nMALS = 2, int nOverlap = 1,
              bool useTTgmres = false, int gmresMaxIter = 25, T gmresRelTol = T(1.e-4));

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_solve_mals_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_SOLVE_MALS_HPP
