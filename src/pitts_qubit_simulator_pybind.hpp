/*! @file pitts_qubit_simulator_pybind.hpp
* @brief header for python binding for PITTS::QubitSimulator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
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
