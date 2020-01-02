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
#include "pitts_fixed_tensortrain.hpp"
#include "pitts_fixed_tensortrain_dot.hpp"
#include "pitts_fixed_tensor3_combine.hpp"
#include "pitts_fixed_tensor3_split.hpp"


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

      //! Setup a new simulator with zero qubits
      //!
      //! @param randomSeed random number generator seed to allow reproducible results
      //!
      QubitSimulator(unsigned int randomSeed = std::mt19937::default_seed) : randomGenerator_(randomSeed) {}

      //! add a new qubit
      //!
      //! @param id   unique qubit id
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
      //! @param id   unique qubit id
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
          swapDims(iDim, iDim+1);
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
      //! @param id   unique qubit id
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
      //! @param id   unique qubit id
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
      void collapseWavefunction(const std::vector<unsigned int>& ids, const std::vector<bool>& values, double tol = 1.e-12)
      {
        if( ids.size() != values.size() )
          throw std::invalid_argument("QubitSimulator::collapseWavefunction: arguments must have the same size!");

        for(int iQ = 0; iQ < ids.size(); iQ++)
        {
          const auto iDim = qubitMap_.at(ids[iQ]);
          auto& subT = stateVec_.editableSubTensors()[iDim];
          const bool up = !values[iQ];
          // set other bit to false and renormalize
          double nrm = 0;
          for(Index j = 0; j < subT.r2(); j++)
            for(Index i = 0; i < subT.r1(); i++)
            {
              if( up )
              {
                nrm += abs2(subT(i,0,j));
                subT(i,1,j) = 0.;
              }
              else // down
              {
                subT(i,0,j) = 0.;
                nrm += abs2(subT(i,1,j));
              }
            }
          nrm = std::sqrt(nrm);
          if( nrm < tol )
            throw std::invalid_argument("QubitSimulator::collapseWavefunction: cannot calculate classical result for given output with zero probability!");
          const double scale = 1./nrm;
          for(Index j = 0; j < subT.r2(); j++)
            for(Index i = 0; i < subT.r1(); i++)
            {
              subT(i,0,j) *= scale;
              subT(i,1,j) *= scale;
            }
        }
      }

      //! Measure a set of Qubits and returns their value
      //!
      //! Collapses the wave function for those qubits.
      //!
      //! @param ids    qubit ids to measure
      //! @return       vector with resulting values ("down" is interpreted as true)
      //!
      std::vector<bool> measureQubits(const std::vector<unsigned int> ids)
      {
        // sub-optimal variant with 2^N runtime
        const auto nQ = stateVec_.nDims();
        const auto rnd = uniformDistribution_(randomGenerator_);

        FixedTensorTrain<std::complex<double>,2> unitVec(nQ);
        std::vector<int> unitVecIdx(nQ);
        std::complex<double> p = 0;
        std::size_t pick;
        for(pick = 0; pick+1 < (1UL << nQ); pick++)
        {
          for(int i = 0; i < nQ; i++)
            unitVecIdx[i] = pick & (1UL << i);
          unitVec.setUnit(unitVecIdx);
          p += dot(stateVec_, unitVec);
          if( abs2(p) > rnd*rnd )
            break;
        }

        std::vector<bool> result;
        for(const auto id: ids)
        {
          bool down = pick & (1UL << qubitMap_.at(id));
          result.push_back(down);
        }

        collapseWavefunction(ids, result);

        return result;
      }

    private:
      //! helper type for indexing qubits in the state vector
      using Index = int;

      //! Current state of the (possibly entangled) qubits
      //!
      //! Represents the 2^N dimensional wave-function in a tensor train format where each sub-tensor represents one qubit.
      //!
      FixedTensorTrain<std::complex<double>,2> stateVec_{0};

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
      void swapDims(Index i, Index j)
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
  };
}


#endif // PITTS_QUBIT_SIMULATOR_HPP
