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
#include <jlcxx/stl.hpp>
#include "pitts_qubit_simulator.hpp"
#include "pitts_qubit_simulator_jlbind.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for Julia bindings
  namespace jlbind
  {
   template<typename T>
   using Matrix2Base = std::array<std::array<T,2>,2>;
   template<typename T>
   struct Matrix2: public Matrix2Base<T> {};
   template<typename T>
   using Matrix4Base = std::array<std::array<T,4>,4>;
   template<typename T>
   struct Matrix4: public Matrix4Base<T> {};

   using ComplexMatrix2 = Matrix2<std::complex<double>>;
   using ComplexMatrix4 = Matrix4<std::complex<double>>;

    // create jlcxx-wrapper for PITTS::QubitSimulator
    void define_QubitSimulator(jlcxx::Module& m)
    {
      m.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("Matrix2", jlcxx::julia_type("DenseMatrix"))
        .apply<ComplexMatrix2>([](auto wrapped)
        {
          wrapped.module().set_override_module(jl_base_module);
          wrapped.template method("size", [](const ComplexMatrix2&){return std::tuple<jlcxx::cxxint_t,jlcxx::cxxint_t>(2,2);});
          wrapped.template method("getindex", [](const ComplexMatrix2& m2, jlcxx::cxxint_t i, jlcxx::cxxint_t j) -> std::complex<double> {return m2[i-1][j-1];});
          wrapped.template method("setindex!", [](ComplexMatrix2& m2, std::complex<double> val, jlcxx::cxxint_t i, jlcxx::cxxint_t j){m2[i-1][j-1] = val;});
          wrapped.module().unset_override_module();
        });

      m.add_type<jlcxx::Parametric<jlcxx::TypeVar<1>>>("Matrix4", jlcxx::julia_type("DenseMatrix"))
        .apply<ComplexMatrix4>([](auto wrapped)
        {
          wrapped.module().set_override_module(jl_base_module);
          wrapped.template method("size", [](const ComplexMatrix4&){return std::tuple<jlcxx::cxxint_t,jlcxx::cxxint_t>(4,4);});
          wrapped.template method("getindex", [](const ComplexMatrix4& m4, jlcxx::cxxint_t i, jlcxx::cxxint_t j) -> std::complex<double> {return m4[i-1][j-1];});
          wrapped.template method("setindex!", [](ComplexMatrix4& m4, std::complex<double> val, jlcxx::cxxint_t i, jlcxx::cxxint_t j){m4[i-1][j-1] = val;});
          wrapped.module().unset_override_module();
        });


      m.add_type<QubitSimulator>("QubitSimulator")
        .constructor<unsigned int>()
        .method("allocate_qubit", &QubitSimulator::allocateQubit)
        .method("deallocate_qubit", &QubitSimulator::deallocateQubit)
        .method("is_classical", &QubitSimulator::isClassical)
        .method("get_classical_value", &QubitSimulator::getClassicalValue)
        .method("collapse_wavefunction", &QubitSimulator::collapseWavefunction)
        .method("measure_qubits", &QubitSimulator::measureQubits)
        .method("get_probability", &QubitSimulator::getProbability)
        .method("apply_single_qubit_gate", [](QubitSimulator& sim, QubitSimulator::QubitId id, const ComplexMatrix2& M)
        {
          sim.applySingleQubitGate(id, M);
        })
        .method("apply_two_qubit_gate", [](QubitSimulator &sim, QubitSimulator::QubitId i, QubitSimulator::QubitId j, const ComplexMatrix4& M)
        {
          sim.applyTwoQubitGate(i, j, M);
        })
        .method("emulate_time_evolution", [](QubitSimulator &sim, double time, const std::vector<QubitSimulator::QubitId>& ids, const std::vector<ComplexMatrix2>& terms, double accuracy = 1.e-12)
        {
          std::vector<QubitSimulator::Matrix2> terms_(terms.size());
          for(int i = 0; i < terms.size(); i++)
            terms_[i] = terms[i];
          sim.emulateTimeEvolution(time, ids, terms_, accuracy);
        })
        .method("get_expectation_value", [](QubitSimulator& sim, const std::vector<QubitSimulator::QubitId>& ids, const std::vector<ComplexMatrix2>& terms)
        {
          std::vector<QubitSimulator::Matrix2> terms_(terms.size());
          for(int i = 0; i < terms.size(); i++)
            terms_[i] = terms[i];
          return sim.getExpectationValue(ids, terms_);
        })
        .method("get_tt_ranks",
            [](const QubitSimulator& qsim){return qsim.getWaveFunction().getTTranks();});
    }
  }
}
