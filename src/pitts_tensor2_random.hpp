// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor2_random.hpp
* @brief fill simple rank-2 tensor with random values
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
*
**/

// include guard
#ifndef PITTS_TENSOR2_RANDOM_HPP
#define PITTS_TENSOR2_RANDOM_HPP

// includes
#include "pitts_tensor2.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! fill a rank-2 tensor with random values
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  void randomize(Tensor2View<T>& t2);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensor2_random_impl.hpp"
#endif

#endif // PITTS_TENSOR2_RANDOM_HPP
