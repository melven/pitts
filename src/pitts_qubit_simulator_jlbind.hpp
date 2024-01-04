// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_qubit_simulator_jlbind.hpp
* @brief header for Julia binding for PITTS::QubitSimulator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-12-18
*
**/


// include guard
#ifndef PITTS_QUBIT_SIMULATOR_JLBIND_HPP
#define PITTS_QUBIT_SIMULATOR_JLBIND_HPP

// includes
#include <jlcxx/jlcxx.hpp>


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for Julia bindings
  namespace jlbind
  {
    //! create jlcxx-wrapper for PITTS::QubitSimulator
    void define_QubitSimulator(jlcxx::Module& mod);
  }
}

#endif // PITTS_QUBIT_SIMULATOR_JLBIND_HPP
