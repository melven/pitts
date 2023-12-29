// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_qubit_simulator_jlbind.cpp
* @brief Julia binding for PITTS::QubitSimulator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-12-18
*
**/

// includes
#include <jlcxx/jlcxx.hpp>
#include "pitts_qubit_simulator.hpp"
#include "pitts_qubit_simulator_jlbind.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for Julia bindings
  namespace jlbind
  {
    // create jlcxx-wrapper for PITTS::QubitSimulator
    void define_QubitSimulator(jlcxx::Module& m)
    {
      m.add_type<QubitSimulator>("QubitSimulator")
        .constructor<unsigned int>()
        .method("allocate_qubit", &QubitSimulator::allocateQubit)
        .method("deallocate_qubit", &QubitSimulator::deallocateQubit)
        .method("is_classical", &QubitSimulator::isClassical)
        .method("get_classical_value", &QubitSimulator::getClassicalValue)
        .method("collapse_wavefunction", &QubitSimulator::collapseWavefunction)
        .method("measure_qubits", &QubitSimulator::measureQubits)
        .method("get_probability", &QubitSimulator::getProbability)
        .method("apply_single_qubit_gate", &QubitSimulator::applySingleQubitGate)
        .method("apply_two_qubit_gate", &QubitSimulator::applyTwoQubitGate)
        .method("emulate_time_evolution", &QubitSimulator::emulateTimeEvolution)
        .method("get_expectation_value", &QubitSimulator::getExpectationValue)
        .method("get_tt_ranks",
            [](const QubitSimulator& qsim){return qsim.getWaveFunction().getTTranks();});
    }
  }
}
