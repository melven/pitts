// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor2_jlbind.hpp
* @brief header for Julia binding for PITTS::Tensor2
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-12-29
*
**/


// include guard
#ifndef PITTS_TENSOR2_JLBIND_HPP
#define PITTS_TENSOR2_JLBIND_HPP

// includes
#include <jlcxx/jlcxx.hpp>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for Julia bindings
  namespace jlbind
  {
    //! create jlcxx-wrapper for PITTS::Tensor2
    void define_Tensor2(jlcxx::Module& m);
  }
}

#endif // PITTS_TENSOR2_JLBIND_HPP
