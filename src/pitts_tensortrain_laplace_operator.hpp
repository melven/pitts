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
#include <cmath>
#include "pitts_tensortrain.hpp"

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
  T laplaceOperator(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()));

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_laplace_operator_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_LAPLACE_OPERATOR_HPP
