// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_eigen.hpp
* @brief Helper file to include the C++ library Eigen (https://eigen.tuxfamily.org)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-01-15
*
**/

// include guard
#ifndef PITTS_EIGEN_HPP
#define PITTS_EIGEN_HPP

// include the content of Eigen/Dense individually, so we can fix the optimization options for Eigen/SVD (which does not work with unsafe-math-optimizations)
//#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/QR>

// use Eigen/SVD but avoid unsupported compiler optimizations...
#ifndef EIGEN_USE_LAPACKE
#  if defined(__INTEL_COMPILER) || defined(__clang__)
#    pragma float_control(precise, on, push)
#  else
#    pragma GCC push_options
#    pragma GCC optimize("no-unsafe-math-optimizations")
#  endif
#endif

#include <Eigen/SVD>

#ifndef EIGEN_USE_LAPACKE
#  if defined(__INTEL_COMPILER) || defined(__clang__)
#    pragma float_control(pop)
#  else
#    pragma GCC pop_options
#  endif
#endif

#include <Eigen/Eigenvalues>
// for ALIGNMENT
#include "pitts_chunk.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! global alignment of PITTS::Chunk expressed for Eigen
  constexpr auto EigenAligned = []()
  {
    if constexpr ( ALIGNMENT % 128 == 0 )
      return Eigen::Aligned128;
    else
      return Eigen::Aligned64;
    static_assert(ALIGNMENT % 64 == 0);
  }();
}

#endif // PITTS_EIGEN_HPP
