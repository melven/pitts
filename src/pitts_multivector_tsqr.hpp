// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_tsqr.hpp
* @brief calculate the QR-decomposition of a multi-vector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-07-13
*
* Based on
* Demmel et.al.: "Communication-optimal Parallel and Sequential QR and LU Factorizations", SISC 2012, doi 10.1137/080731992
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_TSQR_HPP
#define PITTS_MULTIVECTOR_TSQR_HPP

// includes
#include "pitts_multivector.hpp"
#include "pitts_tensor2.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Calculate R from a QR decomposition of a multivector M
  //!
  //! MPI+OpenMP parallel TSQR implementation that calculates ony R. Q is built implicitly but never stored to reduce memory transfers.
  //! It is based on Householder reflections, robust and rank-preserving (just returns rank-deficient R if M does not have full rank).
  //!
  //! @tparam T underlying data type
  //!
  //! @param M                input matrix, possibly distributed over multiple MPI processes
  //! @param R                output matrix R of a QR decomposition of M (with the same singular values and right-singular vectors as M)
  //! @param reductionFactor  (performance-tuning parameter) defines the #chunks of the work-array in the TSQR reduction;
  //!                         set to zero to let this function choose a suitable value automatically
  //! @param mpiGlobal        perform a reduction of R over all MPI processes? (if false, each MPI process does its individual QR decomposition)
  //! @param colBlockingSize  (performance-tuning parameter) used for improving data accesses when m is large
  //!                         set to zero to let this function choose a suitable value automatically
  //!
  template<typename T>
  void block_TSQR(const MultiVector<T>& M, Tensor2<T>& R, int reductionFactor = 0, bool mpiGlobal = true, int colBlockingSize = 0);
  
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_tsqr_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_TSQR_HPP
