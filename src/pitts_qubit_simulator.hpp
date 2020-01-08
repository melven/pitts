/*! @file pitts_qubit_simulator.hpp
* @brief simplistic simulator backend for a gate-based quantum computer
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-30
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_QUBIT_SIMULATOR_HPP
#define PITTS_QUBIT_SIMULATOR_HPP

// includes
#include <unordered_map>
#include <complex>
#include <random>
#include <exception>
#include <array>
#include "pitts_fixed_tensortrain.hpp"
#include "pitts_fixed_tensortrain_dot.hpp"
#include "pitts_fixed_tensortrain_axpby.hpp"
#include "pitts_fixed_tensor3_combine.hpp"
#include "pitts_fixed_tensor3_split.hpp"
#include "pitts_fixed_tensor3_apply.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // simple helper functions
  namespace
  {
    //! return the square of the absolute of a complex number
    constexpr double abs2(std::complex<double> c)
    {
      return std::real(c)*std::real(c) + std::imag(c)*std::imag(c);
    }
  }


  //! Simplistic backend for simulating a gate-based quantum computer
  class QubitSimulator
  {
    public:
      //! Unique number for identifying a Qubit
      using QubitId = unsigned int;

      //! Data type for single Qubit gates (2x2 matrix)
      using Matrix2 = std::array<std::array<std::complex<double>,2>,2>;

      //! Data type for two-Qubit gates (4x4 matrix)
      using Matrix4 = std::array<std::array<std::complex<double>,4>,4>;

      //! Data type for storing the state of the qubits
      using StateVector = FixedTensorTrain<std::complex<double>,2>;

      //! Setup a new simulator with zero qubits
      //!
      //! @param randomSeed random number generator seed to allow reproducible results
      //!
      QubitSimulator(unsigned int randomSeed = std::mt19937::default_seed) : randomGenerator_(randomSeed) {}

      //! add a new qubit
      //!
      //! @param id   unique identifier for the qubit
      //!
      void allocateQubit(QubitId id)
      {
        const Index nQubits = stateVec_.nDims();
        const auto [ignored, inserted] = qubitMap_.insert({id, nQubits});
        if( !inserted )
          throw std::invalid_argument("QubitSimulator::allocateQubit: a qubit with the given ID already exists!");
        stateVec_.editableSubTensors().resize(nQubits+1);
        qubitIds_.push_back(id);
        // new qubit goes up
        stateVec_.editableSubTensors().back().resize(1,1);
        stateVec_.editableSubTensors().back()(0,0,0) = 1.;
        stateVec_.editableSubTensors().back()(0,1,0) = 0.;
        /*
        */
      }

      //! remove a qubit
      //!
      //! @warning the qubit must be in a "classical" state, e.g. after a measurement.
      //!
      //! @param id   specifies the desired qubit by its id
      //!
      void deallocateQubit(QubitId id)
      {
        if( !isClassical(id) )
          throw std::invalid_argument("QubitSimulator::deallocateQubit: Qubit has not been measured, cannot deallocate it!");

        // id exists, otherwise is_classical would already have thrown an error!
        Index iDim = qubitMap_[id];
        // move dimension to the end of the state vector
        while( iDim != stateVec_.nDims()-1 )
        {
          swapDims_(iDim, iDim+1);
          iDim++;
        }
        // we need to remove the last sub-tensor now... last TT-rank should be one!
        if( stateVec_.subTensors()[iDim].r1() != 1 )
          throw std::runtime_error("QubitSimulator::deallocateQubit: unexpected dimensions, internal error!");
        stateVec_.editableSubTensors().pop_back();
        qubitIds_.pop_back();
        qubitMap_.erase(id);
      }

      //! check if a qubit is in classical (not a superposition) state
      //!
      //! @param id   specifies the desired qubit by its id
      //! @param tol  floating point tolerance for detecting a non-zero probability
      //!
      bool isClassical(QubitId id, double tol = 1.e-10)
      {
        const Index iDim = qubitMap_.at(id);

        bool up = false, down = false;
        const auto& subT = stateVec_.subTensors()[iDim];
        for(Index i = 0; i < subT.r1(); i++)
          for(Index j = 0; j < subT.r2(); j++)
          {
            if( abs2(subT(i,0,j)) > tol*tol )
              up = true;
            if( abs2(subT(i,1,j)) > tol*tol )
              down = true;
          }

        return up^down;
      }

      //! get the value of a qubit in classical state
      //!
      //! @warning the qubit must be in a "classical" state, e.g. after a measurement.
      //!
      //! @param id   specifies the desired qubit by its id
      //! @param tol  floating point tolerance for detecting a non-zero probability
      //! @return     value of the qubit: "up" is interpreted as false and "down" as true.
      //!
      bool getClassicalValue(QubitId id, double tol = 1.e-10)
      {
        const Index iDim = qubitMap_.at(id);

        bool up = false, down = false;
        const auto& subT = stateVec_.subTensors()[iDim];
        for(Index j = 0; j < subT.r2(); j++)
          for(Index i = 0; i < subT.r1(); i++)
          {
            if( abs2(subT(i,0,j)) > tol*tol )
              up = true;
            if( abs2(subT(i,1,j)) > tol*tol )
              down = true;
          }

        if( up && down )
          throw std::invalid_argument("QubitSimulator::getClassicalValue: Qubit has not been measured, cannot access its classical value!");

        return down;
      }

      //! Set qubits to the given classical state
      //!
      //! @param ids    qubit ids that have been measured -> wave function collapses
      //! @param values measured values of the qubits ("down" is interpreted as true)
      //! @param tol    numerical tolerance for rejecting an impossible outcome (throws an error)
      //!
      void collapseWavefunction(const std::vector<QubitId>& ids, const std::vector<bool>& values, double tol = 1.e-12)
      {
        if( ids.size() != values.size() )
          throw std::invalid_argument("QubitSimulator::collapseWavefunction: arguments must have the same size!");

        // nothing happens for empty input
        if( ids.empty() )
          return;

        StateVector newState(stateVec_.nDims());
        projectStateVec_(ids, values, newState);

        // check if result is possible (todo: add missing norm function)
        const double nrm = std::sqrt(std::real(dot(newState, newState)));
        if( nrm < tol )
          throw std::invalid_argument("QubitSimulator::collapseWavefunction: cannot calculate classical result for given output with zero probability!");

        // renormalize
        for(Index iDim = 0; iDim+1 < newState.nDims(); iDim++)
        {
          auto& subT1 = newState.editableSubTensors()[iDim];
          auto& subT2 = newState.editableSubTensors()[iDim+1];
          const auto subT12 = combine(subT1, subT2);
          split(subT12, subT1, subT2);
        }
        // we must scale the last subtensor now
        const auto invNrm = 1./nrm;
        {
          auto& lastSubT = newState.editableSubTensors().back();
          for(int j = 0; j < lastSubT.r2(); j++)
            for(int i = 0; i < lastSubT.r1(); i++)
            {
              lastSubT(i,0,j) *= invNrm;
              lastSubT(i,1,j) *= invNrm;
            }
        }

        std::swap(newState, stateVec_);
      }

      //! Measure a set of Qubits and returns their value
      //!
      //! Collapses the wave function for those qubits.
      //!
      //! @param ids    qubit ids to measure
      //! @return       vector with resulting values ("down" is interpreted as true)
      //!
      std::vector<bool> measureQubits(const std::vector<QubitId>& ids)
      {
        // evaluate one qubit at a time => avoids 2^N runtime!
        std::vector<bool> result;
        StateVector tmp(stateVec_.nDims());
        for(auto id: ids)
        {
          // calculate the probability for this qubit pointing down
          projectStateVec_({id}, {true}, tmp);
          double probability = std::real(dot(tmp, tmp));

          // choose randomly up/down
          if( uniformDistribution_(randomGenerator_) < probability )
          {
            result.push_back(true);
          }
          else
          {
            result.push_back(false);
            // calculate new projection
            projectStateVec_({id}, {false}, tmp);
            probability = 1 - probability;
          }
          // rescale (collapse but without re-normalization)
          {
            const auto invNrm = 1 / std::sqrt(probability);
            const auto iDim = qubitMap_.at(id);
            auto& subT = tmp.editableSubTensors()[iDim];
            for(int j = 0; j < subT.r2(); j++)
              for(int i = 0; i < subT.r1(); i++)
              {
                subT(i,0,j) *= invNrm;
                subT(i,1,j) *= invNrm;
              }
          }

          std::swap(stateVec_, tmp);
        }

        // renormalize
        for(Index iDim = 0; iDim+1 < stateVec_.nDims(); iDim++)
        {
          auto& subT1 = stateVec_.editableSubTensors()[iDim];
          auto& subT2 = stateVec_.editableSubTensors()[iDim+1];
          const auto subT12 = combine(subT1, subT2);
          split(subT12, subT1, subT2);
        }

        return result;
      }

      //! Calculate the probability of a given outcome when measuring the specified qubits
      //!
      //! @param ids    qubit ids to analyze
      //! @param values desired values of the qubits ("down" is interpreted as true)
      //!
      double getProbability(const std::vector<QubitId>& ids, const std::vector<bool>& values)
      {
        StateVector dummyState(stateVec_.nDims());
        projectStateVec_(ids, values, dummyState);

        // todo: add missing norm function
        return std::real(dot(dummyState, dummyState));
      }

      //! Apply a quantum gate to a single qubit (without control qubits)
      //!
      //! @param id   desired qubit id
      //! @param M    2x2 matrix that represents the desired gate, must be orthogonal
      //!
      void applySingleQubitGate(QubitId id, const Matrix2& M)
      {
        const Index iDim = qubitMap_.at(id);
        auto& subT = stateVec_.editableSubTensors()[iDim];
        apply(subT, M);
      }

      //! Apply a quantum gate to a pair of qubits
      //!
      //! @param id1  specifies the first qubit in the gate
      //! @param id2  specifies the second qubit in the gate
      //! @param M    4x4 matrix that represents the desired gate, must be orthogonal
      //!
      void applyTwoQubitGate(QubitId id1, QubitId id2, const Matrix4& M)
      {
        auto iDim1 = qubitMap_.at(id1);
        auto iDim2 = qubitMap_.at(id2);

        // move dimensions by swapping until iDim2 = iDim1+1
        makeNeighborDims_(iDim1, iDim2);

        // apply gate to combined tensor...
        auto& subT1 = stateVec_.editableSubTensors()[iDim1];
        auto& subT2 = stateVec_.editableSubTensors()[iDim2];
        auto subT12 = combine(subT1, subT2);
        apply(subT12, M);
        split(subT12, subT1, subT2);
      }

      //! Apply exp(i*time*H) to the state (wave function), i. e. evolve under the Hamiltonian H for a given time
      //!
      //! The Hamiltonian H consists of 2x2 matrices for different qubits: All 2x2 matrices must hermitian (self-adjoint)
      //! so expm(i*time*H) is an orthogonal operator.
      //!
      //! @warning This currently assumes that the operator has approximately norm 1...
      //!
      //! @param time     desired simulated time, can be negative (operations are reversible, so we can both go forward and backward
      //! @param ids      specifies the qubit for each term
      //! @param terms    composition of the Hamiltonian of the complete system into real, symmetric 2x2 matrices for individual qubits
      //! @param accuracy desired numerical tolerance for the computation
      //!
      void emulateTimeEvolution(double time, const std::vector<QubitId>& ids, const std::vector<Matrix2>& terms, double accuracy = 1.e-12)
      {
        // simplistic explicit RK4 approach

        // first calculate a suitable timestep size
        const int nSteps = int( std::abs(time) / std::pow(accuracy, 0.25) + 1 );
        const double dt = time / nSteps;

        // helper variables
        StateVector dx1(stateVec_.nDims());
        StateVector dx2(stateVec_.nDims());
        StateVector dx3(stateVec_.nDims());
        StateVector dx4(stateVec_.nDims());

        static constexpr auto i = std::complex<double>(0, 1);
        static constexpr auto one = std::complex<double>(1, 0);

        for(int iStep = 0; iStep < nSteps; iStep++)
        {
          dx1 = stateVec_;
          applyTerms_(dx1, ids, terms);

          dx2 = stateVec_;
          axpby(i*dt/2., dx1, one, dx2);
          applyTerms_(dx2, ids, terms);

          dx3 = stateVec_;
          axpby(i*dt/2., dx2, one, dx3);
          applyTerms_(dx3, ids, terms);

          dx4 = stateVec_;
          axpby(i*dt, dx3, one, dx4);
          applyTerms_(dx4, ids, terms);

          axpby(i*dt/6., dx1, one, stateVec_);
          axpby(i*dt/3., dx2, one, stateVec_);
          axpby(i*dt/3., dx3, one, stateVec_);
          axpby(i*dt/6., dx4, one, stateVec_);
        }
      }

      //! Get the expectation value of a qubit operator w.r.t. the current wave function
      //!
      //! This calculates <Psi|H|Psi> where Psi is the current state (wave function) and the operator H is composed of a sum of single-qubit 2x2 matrices.
      //!
      //! @param ids      specifies the qubit for each term
      //! @param terms    composes the desired operator as a sum of single-qubit operators
      //!
      double getExpectationValue(const std::vector<QubitId>& ids, const std::vector<Matrix2>& terms)
      {
        if( ids.size() != terms.size() )
          throw std::invalid_argument("QubitSimulator::getExpectationValue: input arrays have different size!");

        double result = 0;
        auto tmpState = stateVec_;
        for(int iTerm = 0; iTerm < terms.size(); iTerm++)
        {
          const Index iDim = qubitMap_.at(ids[iTerm]);
          auto& subT = tmpState.editableSubTensors()[iDim];
          apply(subT, terms[iTerm]);
          result += std::real(dot(stateVec_, tmpState));

          // reset temporary state so we can apply another term of the operator in the next iteration
          subT = stateVec_.subTensors()[iDim];
        }
        return result;
      }

      //! current state (for testing / debugging purposes)
      const auto& getWaveFunction() const {return stateVec_;}

      //! current map of qubit ids to the dimension index in the state (for testing / debugging purposes)
      const auto& getQubitMap() const {return qubitMap_;}

    private:
      //! helper type for indexing qubits in the state vector
      using Index = int;

      //! Current state of the (possibly entangled) qubits
      //!
      //! Represents the 2^N dimensional wave-function in a tensor train format where each sub-tensor represents one qubit.
      //!
      StateVector stateVec_{0};

      //! mapping of qubits ids to their sub-tensor position in the stateVec_
      //!
      //! This allows using arbitrary ids for identifying a qubit as well as reordering the dimensions of the stateVec_ tensor.
      //!
      std::unordered_map<QubitId, Index> qubitMap_;

      //! mapping of qubit sub-tensor position to qubit id (inverse of qubitMap_)
      std::vector<QubitId> qubitIds_;

      //! random number generator
      std::mt19937 randomGenerator_;

      //! random number distribution
      std::uniform_real_distribution<double> uniformDistribution_{0., 1.};

      //! helper function to modify the ordering of dimensions in the stateVec_
      void swapDims_(Index i, Index j)
      {
        if( i+1 != j )
          throw std::invalid_argument("QubitSimulator: can only swap neighboring dimensions!");

        auto& subT1 = stateVec_.editableSubTensors().at(i);
        auto& subT2 = stateVec_.editableSubTensors().at(j);

        // combine sub-tensors with transposed order and split again
        const auto subT12 = combine(subT1, subT2, true);
        split(subT12, subT1, subT2);

        // also adjust ids
        const QubitId iQ = qubitIds_.at(i);
        const QubitId jQ = qubitIds_.at(j);
        std::swap(qubitMap_.at(iQ), qubitMap_.at(jQ));
        std::swap(qubitIds_.at(i), qubitIds_.at(j));
      }

      //! helper function to move two dimensions next to each other (using swapDims)
      //!
      //! @param[inout] i   first qubit index, new index of the same qubit (same id) on output
      //! @param[inout] n   second qubit index, new index of the same qubit (same id) on output
      //!
      void makeNeighborDims_(Index& i, Index& j)
      {
        if( i == j || i < 0 || j < 0 || i >= stateVec_.nDims() || j >= stateVec_.nDims() )
          throw std::runtime_error("QubitSimulator: invalid indices!");

        while( j != i+1 )
        {
          if( i < j )
          {
            swapDims_(i, i+1);
            i++;
            continue;
          }
          if( i == j+1 )
          {
            swapDims_(j, i);
            std::swap(j, i);
            continue;
          }
          // i > j+1
          swapDims_(j, j+1);
          j++;
        }
      }

      //! helper function to project the stateVec_ onto a given result (un-normalized collapse)
      //!
      //! @param[in]  ids            qubits to consider
      //! @param[in]  values         desired values of the qubits ("down" is interpreted as true)
      //! @param[out] projectedState resulting projected state vector
      //!
      void projectStateVec_(const std::vector<QubitId>& ids, const std::vector<bool>& values, StateVector& projectedState) const
      {
        if( ids.size() != values.size() )
          throw std::invalid_argument("QubitSimulator::projectStateVec: arguments must have the same size!");

        projectedState = stateVec_;

        for(int iQ = 0; iQ < ids.size(); iQ++)
        {
          const Index iDim = qubitMap_.at(ids[iQ]);
          auto& subT = projectedState.editableSubTensors()[iDim];
          const bool up = !values[iQ];
          // set other bit to false
          for(Index j = 0; j < subT.r2(); j++)
            for(Index i = 0; i < subT.r1(); i++)
            {
              if( up )
                subT(i,1,j) = 0.;
              else // down
                subT(i,0,j) = 0.;
            }
        }
      }

      //! helper function to apply a bunch of single qubit transformations at once
      void applyTerms_(StateVector& state, const std::vector<QubitId>& ids, const std::vector<Matrix2>& terms) const
      {
        if( ids.size() != terms.size() )
          throw std::invalid_argument("QubitSimulator::applyTerms: number of ids and terms mismatch!");

        for(int iQ = 0; iQ < ids.size(); iQ++)
        {
          const Index iDim = qubitMap_.at(ids[iQ]);
          auto& subT = state.editableSubTensors()[iDim];
          apply(subT, terms[iQ]);
        }
      }
  };
}


#endif // PITTS_QUBIT_SIMULATOR_HPP
