// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_common_impl.hpp
* @brief Useful helper functionality, e.g. MPI/OpenMP initialization
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-25
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
#include "pitts_chunk_ops.hpp"
#include "pitts_eigen.hpp"
#include "pitts_machine_info.hpp"

//#ifndef EIGEN_USE_LAPACKE
//#include <immintrin.h>
//#endif

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
    }

    if( MPI_Initialized(&common::mpiInitializedBefore) != 0 )
      throw std::runtime_error("MPI error");

    if( !common::mpiInitializedBefore )
      if( MPI_Init(argc, argv) != 0 )
        throw std::runtime_error("MPI error");

    // initialize the machine information like cache sizes...
    MachineInfo mi = getMachineInfo(true);

    const auto& [iProc, nProcs] = internal::parallel::mpiProcInfo();
    if( iProc == 0 && verbose )
    {
#ifdef PITTS_GIT_VERSION
      std::cout << "PITTS: version " << PITTS_GIT_VERSION << "\n";
#endif

      std::cout << "PITTS: MPI #procs: " << nProcs << "\n";
#pragma omp parallel
      {
        const auto& [iThread, nThreads] = internal::parallel::ompThreadInfo();
        if( iThread == 0 )
          std::cout << "PITTS: OpenMP #threads: " << nThreads << "\n";
      }
      std::cout << "PITTS: SIMD implementation: " << SIMD_implementation() << "\n";
      std::cout << "PITTS: Eigen SIMD implementation: " << Eigen::SimdInstructionSetsInUse() << "\n";

      std::cout << "PITTS: cache sizes (L1/core, L2/core, L3/accessible) [byte]: (" << mi.cacheSize_L1_perCore << ", " << mi.cacheSize_L2_perCore << ", " << mi.cacheSize_L3_total << ")\n";

      // more information on preprocessor directives used during compilation
      std::cout << "PITTS: preprocessor definitions:"
#ifdef PITTS_DEVELOP_BUILD
                << " PITTS_DEVELOP_BUILD"
#endif
#ifdef PITTS_TENSORTRAIN_PLAIN_AXPBY
                << " PITTS_TENSORTRAIN_PLAIN_AXPBY"
#endif
#ifdef PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB
                << " PITTS_TENSORTRAIN_NORMALIZE_PLAIN_QB"
#endif
#ifdef PITTS_TENSORTRAIN_MALS_SLOWCONTRACT
                << " PITTS_TENSORTRAIN_MALS_SLOWCONTRACT"
#endif
#ifdef PITTS_DIRECT_MKL_GEMM
                << " PITTS_DIRECT_MKL_GEMM"
#endif
                << "\n";
    }

#ifdef PITTS_USE_LIKWID_MARKER_API
    LIKWID_MARKER_INIT;
#pragma omp parallel
    {
      LIKWID_MARKER_THREADINIT;
    }
#endif

    // seed random number generator
    internal::randomGenerator.seed(randomSeed);

//    // as a workaround for problems with Eigen::BDCSVD (https://gitlab.com/libeigen/eigen/-/issues/2663)
//    // disable flush-to-zero mode
//#ifndef EIGEN_USE_LAPACKE
//#pragma omp parallel
//    {
//      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
//      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
//    }
//#endif
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
