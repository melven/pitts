/*! @file pitts_common.hpp
* @brief Useful helper functionality, e.g. MPI/OpenMP initialization
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-25
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_COMMON_HPP
#define PITTS_COMMON_HPP

// includes
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include "pitts_performance.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // internal namespace for helper variables
  namespace common
  {
    //! flag to indicate if PITTS called MPI_Init or somebody else did it before
    static inline int mpiInitializedBefore = false;
  }


  //! Call MPI_Init if needed and print some general informations
  //!
  //! @param argc     pointer to main argc argument for MPI_Init
  //! @param argv     pointer to main argv argument for MPI_Init
  //! @param verbose  set to false to omit any console output (info on #threads and #processes)
  //!
  void initialize(int* argc, char** argv[], bool verbose = true)
  {
    // first init OpenMP threads (before MPI, to make it easier to pin)
#pragma omp parallel
    {
      if( omp_get_thread_num() == 0 && verbose )
        std::cout << "PITTS: OpenMP #threads: " << omp_get_num_threads() << "\n";
    }

    if( MPI_Initialized(&common::mpiInitializedBefore) != 0 )
      throw std::runtime_error("MPI error");

    if( !common::mpiInitializedBefore )
      if( MPI_Init(argc, argv) != 0 )
        throw std::runtime_error("MPI error");

    int nProcs = 1, iProc = 0;
    if( MPI_Comm_size(MPI_COMM_WORLD, &nProcs) != 0 )
      throw std::runtime_error("MPI error");
    if( MPI_Comm_rank(MPI_COMM_WORLD, &iProc) != 0 )
      throw std::runtime_error("MPI error");
    if( iProc == 0 && verbose )
      std::cout << "PITTS: MPI #procs: " << nProcs << "\n";
  }


  //! Call MPI_Finalize if needed and print some statistics
  //!
  //! @param verbose set to false to omit any console output (for timing statistics)
  //!
  void finalize(bool verbose = true)
  {
    if( verbose )
      PITTS::performance::printStatistics();

    if( !common::mpiInitializedBefore )
      if( MPI_Finalize() != 0 )
        throw std::runtime_error("MPI error");
  }

}


#endif // PITTS_COMMON_HPP
