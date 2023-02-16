/*! @file pitts_qubit_simulator_pybind.cpp
* @brief python binding for PITTS::QubitSimulator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-01-06
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// includes
#include <variant>
#include "pitts_qubit_simulator.hpp"
#include "pitts_eigen.hpp"
#include "pitts_qubit_simulator_pybind.hpp"

// include pybind11 last (workaround for problem with C++20 modules)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

namespace py = pybind11;


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for python bindings
  namespace pybind
  {
    // create pybind11-wrapper for PITTS::QubitSimulator
    void init_QubitSimulator(py::module& m)
    {
      py::class_<QubitSimulator>(m, "QubitSimulator", "Simplistic backend for simulating a gate-based quantum computer")
        .def(py::init<unsigned int>(), py::arg("randomSeed")=5489, "Create QubitSimulator with given random number generator seed") // mt19937::default_seed
        .def("allocate_qubit", &QubitSimulator::allocateQubit, "Add a new qubit with the given id")
        .def("deallocate_qubit", &QubitSimulator::deallocateQubit, "Remove the qubit with the given id")
        .def("is_classical", &QubitSimulator::isClassical, py::arg("id"), py::arg("tol")=1.e-8, "check if qubit is in classical state")
        .def("get_classical_value", &QubitSimulator::getClassicalValue, py::arg("id"), py::arg("tol")=1.e-8, "return the value of a qubit in classical state")
        .def("collapse_wavefunction", &QubitSimulator::collapseWavefunction, "set the qubits to the given classical state as if this was the result of a measurement")
        .def("measure_qubits", &QubitSimulator::measureQubits, "Perform a (partial) measurement of the given qubits")
        .def("get_probability", &QubitSimulator::getProbability, "Calculate the probability of a given outcome")
        .def("apply_single_qubit_gate", &QubitSimulator::applySingleQubitGate, "Apply a quantum gate to a single qubit")
        .def("apply_two_qubit_gate", &QubitSimulator::applyTwoQubitGate, "Apply a quantum gate to a pair of qubits")
        .def("emulate_time_evolution", &QubitSimulator::emulateTimeEvolution, "Apply exp(i*time*H) to the wave function where H is composed of single-qubit terms")
        .def("get_expectation_value", &QubitSimulator::getExpectationValue, "Calculate <Psi|H|Psi> where Psi is the wave function and H is the sum of single-qubit matrices")
        .def("get_tt_ranks",
            [](const QubitSimulator& qsim){return qsim.getWaveFunction().getTTranks();}, "Return the tensor train ranks (dimension of the underlying data structure for representing the wave function)");
    }
  }
}
