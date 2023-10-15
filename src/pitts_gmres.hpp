// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_gmres.hpp
* @brief Generic iterative solver for linear systems based on GMRES with templated underlying vector and matrix data type
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-13
*
**/

// include guard
#ifndef PITTS_GMRES_HPP
#define PITTS_GMRES_HPP

// includes
#include <string_view>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Iteratively solves a linear system of equations using the GMRES method
  //!
  //! Approximates x with OpA* x = b
  //!
  //! @tparam LinearOperator  type of the linear operator (e.g. matrix), required for calling apply(Op, x, y)
  //! @tparam Vector          type for vectors, must support axpy(alpha, x, y), dot(x,y), axpy_dot(alpha, x, y, z), norm2(x), axpy_norm2(alpha, x, y), scale(alpha,x)
  //! @tparam T               underlying data type (defines the precision)
  //!
  //! @param OpA              linear operator, can be applied to a vector
  //! @param symmetric        specifies if Op is symmetric to avoid some unnecessary operations (this then yields a variant of the MINRES method)
  //! @param b                right-hand side vector
  //! @param x                initial guess on input, approximate solution on output
  //! @param maxIter          maximal number of iterations (no restarts for now)
  //! @param absResTol        absolute residual tolerance: the iteration aborts if the absolute residual norm is smaller than absResTol
  //! @param relResTol        relative residual tolerance: the iteration aborts if the relative residual norm is smaller than relResTol
  //! @param outputPrefix     string to prefix all output about the convergence history
  //! @param verbose          set to true, to print the residual norm in each iteration to std::cout
  //! @return                 current absolute residual norm ||OpA*x - b||_2 and relative residual norm ||OpA*x-b||_2 / ||OpA*x0-b||_2
  //!
  template<typename T, typename LinearOperator, typename Vector>
  std::pair<T,T> GMRES(const LinearOperator& OpA, bool symmetric, const Vector& b, Vector& x, int maxIter, const T& absResTol, const T& relResTol, const std::string_view& outputPrefix = "", bool verbose = false);
}


#ifndef PITTS_DEVELOP_BUILD
#include "pitts_gmres_impl.hpp"
#endif

#endif // PITTS_GMRES_HPP
