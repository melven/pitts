// Copyright (c) 2021 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_operator_jlbind.hpp
* @brief header for Julia binding for PITTS::TensorTrainOperator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-12-18
*
**/


// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_JLBIND_HPP
#define PITTS_TENSORTRAIN_OPERATOR_JLBIND_HPP

// includes
#include <jlcxx/jlcxx.hpp>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for Julia bindings
  namespace jlbind
  {
    //! create jlcxx-wrapper for PITTS::TensorTrainOperator
    void define_TensorTrainOperator(jlcxx::Module& mod);
  }
}

#endif // PITTS_TENSORTRAIN_OPERATOR_JLBIND_HPP
