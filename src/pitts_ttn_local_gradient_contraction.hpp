// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_ttn_local_gradient_contraction.hpp
* @brief calculate local gradient in a tree tensor network (special series of contractions)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-03-02
*
**/

// include guard
#ifndef PITTS_TTN_LOCAL_GRADIENT_CONTRACTION_HPP
#define PITTS_TTN_LOCAL_GRADIENT_CONTRACTION_HPP

// includes
#include "pitts_multivector.hpp"
#include "pitts_tensor3.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Calculate the gradient of the current tensor
  //!
  //! MPI+OpenMP parallel implementation of a series of contractions required for calculating the local gradient in a tree-tensor network (TTN)
  //! method for machine-learning.
  //!
  //! @tparam T underlying data type
  //!
  //! @param[in]  optTensor       current tensor in the TTN to optimize (r_left x r_right x r_top)
  //! @param[in]  envLeft         left environment tensor (n_samples x r_left)
  //! @param[in]  envRight        right environment tensor (n_samples x r_right)
  //! @param[in]  envTop          top environment tensor (n_samples x (n_classes x r_right))
  //! @param[out] gradOptTensor   result gradient tensor (same dimension as optTensor: r_left x r_right x r_top)
  //! @param[in]  mpiParallel     flag to indicate that the gradOptTensor should be summed up over all MPI processes
  //!
  template<typename T>
  void ttn_local_gradient_contract(const Tensor3<T>& optTensor, const MultiVector<T>& envLeft, const MultiVector<T>& envRight,  const MultiVector<T>& envTop, Tensor3<T>& gradOptTensor, bool mpiParallel = false);
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_ttn_local_gradien_contraction_impl.hpp"
#endif

#endif // PITTS_TTN_LOCAL_GRADIENT_CONTRACTION_HPP
