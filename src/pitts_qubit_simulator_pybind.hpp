// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_qubit_simulator_pybind.hpp
* @brief header for python binding for PITTS::QubitSimulator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-26
*
**/


// include guard
#ifndef PITTS_QUBIT_SIMULATOR_PYBIND_HPP
#define PITTS_QUBIT_SIMULATOR_PYBIND_HPP

// includes
#include <pybind11/pybind11.h>


namespace py = pybind11;


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for python bindings
  namespace pybind
  {
    //! create pybind11-wrapper for PITTS::QubitSimulator
    void init_QubitSimulator(py::module& m);
  }
}

#endif // PITTS_QUBIT_SIMULATOR_PYBIND_HPP
