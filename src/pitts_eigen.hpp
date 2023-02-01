/*! @file pitts_eigen.hpp
* @brief Helper file to include the C++ library Eigen (https://eigen.tuxfamily.org)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-01-15
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
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
#  ifdef __INTEL_COMPILER
#    pragma float_control(precise, on, push)
#  else
#    pragma GCC push_options
#    pragma GCC optimize("no-unsafe-math-optimizations")
#  endif
#endif

#include <Eigen/SVD>

#ifndef EIGEN_USE_LAPACKE
#  ifdef __INTEL_COMPILER
#    pragma float_control(pop)
#  else
#    pragma GCC pop_options
#  endif
#endif

#include <Eigen/Eigenvalues>

#endif // PITTS_EIGEN_HPP
