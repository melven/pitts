/*! @file pitts_common_impl.hpp
* @brief Useful helper functionality, e.g. MPI/OpenMP initialization
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-25
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_COMMON_IMPL_HPP
#define PITTS_COMMON_IMPL_HPP

// includes
#include <iostream>
#include "pitts_parallel.hpp"
#include "pitts_common.hpp"
#include "pitts_performance.hpp"

#ifdef PITTS_USE_LIKWID_MARKER_API
#include <likwid.h>
#endif

// workaround for speeding up compile times during development
#ifndef PITTS_DEVELOP_BUILD
#define INLINE inline
#else
#define INLINE
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // internal namespace for helper variables
  namespace common
  {
    //! flag to indicate if PITTS called MPI_Init or somebody else did it before
    static inline int mpiInitializedBefore = true;
  }


  // implement initialize
  INLINE void initialize(int* argc, char** argv[], bool verbose, std::uint_fast64_t randomSeed)
  {
    // first init OpenMP threads (before MPI, to make it easier to pin)
#pragma omp parallel
    {
      const auto& [iThread, nThreads] = internal::parallel::ompThreadInfo();
      if( iThread == 0 && verbose )
        std::cout << "PITTS: OpenMP #threads: " << nThreads << "\n";
    }

    if( MPI_Initialized(&common::mpiInitializedBefore) != 0 )
      throw std::runtime_error("MPI error");

    if( !common::mpiInitializedBefore )
      if( MPI_Init(argc, argv) != 0 )
        throw std::runtime_error("MPI error");

    const auto& [iProc, nProcs] = internal::parallel::mpiProcInfo();
    if( iProc == 0 && verbose )
      std::cout << "PITTS: MPI #procs: " << nProcs << "\n";

#ifdef PITTS_USE_LIKWID_MARKER_API
    LIKWID_MARKER_INIT;
#pragma omp parallel
    {
      LIKWID_MARKER_THREADINIT;
    }
#endif

    // seed random number generator
    internal::randomGenerator.seed(randomSeed);
  }


  // implement finalize
  INLINE void finalize(bool verbose)
  {
#ifdef PITTS_USE_LIKWID_MARKER_API
    LIKWID_MARKER_CLOSE;
#endif

    if( verbose )
      PITTS::performance::printStatistics();

    if( !common::mpiInitializedBefore )
      if( MPI_Finalize() != 0 )
        throw std::runtime_error("MPI error");
  }

}


#endif // PITTS_COMMON_IMPL_HPP