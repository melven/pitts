// Copyright (c) 2022 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_gram_schmidt.hpp
* @brief Modified and iterated Gram-Schmidt for orthogonalizing vectors in TT format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-07-02
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_GRAM_SCHMIDT_HPP
#define PITTS_TENSORTRAIN_GRAM_SCHMIDT_HPP

// includes
#include <limits>
#include <cmath>
#include <vector>
#include <string_view>
#include "pitts_tensortrain.hpp"
#include "pitts_eigen.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Modified Gram-Schmidt orthogonalization algorithm in Tensor-Train format
  //!
  //! Orthogonalizes w wrt V=(v_1, ..., v_k) and normalizes w.
  //! Then adds w to the list of directions V.
  //!
  //! The modified Gram-Schmidt process (MGS) is adopted to the special behavior of tensor-train arithmetic:
  //! Per default uses pivoting (more dot calls), multiple iterations (to increase robustness/accuracy), and skips directions v_i where <v_i,w> is already small.
  //!
  //! @tparam T             data type (double, float, complex)
  //!
  //! @param V                orthogonal directions in TT format, orthogonalized w is appended
  //! @param w                new direction in TT format, orthogonalized wrt. V
  //! @param rankTolerance    desired approximation accuracy (for TT axpby / normalize)
  //! @param maxRank          maximal allowed approximation rank (for TT axpby / normalize)
  //! @param symmetric        set to true for w=Av_k with symmetric operator A to exploit the symmetry (Results in MinRes-like algorithms).
  //! @param outputPrefix     string to prefix all output about the convergence history
  //! @param verbose          set to true, to print the residual norm in each iteration to std::cout
  //! @param nIter            number of iterations for iterated Gram-Schmidt
  //! @param pivoting         enable or disable pivoting (enabled per default)
  //! @param modified         use modified Gram-Schmidt: recalculate <v_i,w> (enabled per default)
  //! @param skipDirs         skip axpy operations, when <v_i,w> already < tolerance
  //! @return                 dot-products <v_1,w> ... <v_k,w> and norm of w after orthog. wrt. V (||w-VV^Tw||)
  //!
  template<typename T>
  Eigen::ArrayX<T> gramSchmidt(std::vector<TensorTrain<T>>& V, TensorTrain<T>& w,
                               T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max(), bool symmetric = false,
                               const std::string_view& outputPrefix = "", bool verbose = false,
                               int nIter = 4, bool pivoting = true, bool modified = true, bool skipDirs = true);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_gram_schmidt_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_GRAM_SCHMIDT_HPP
