// Copyright (c) 2024 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_machine_info_impl.hpp
* @brief Implementation of helper functionality for machine-specific parameters like cache sizes
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2024-07-15
*
**/

// include guard
#ifndef PITTS_MACHINE_INFO_IMPL_HPP
#define PITTS_MACHINE_INFO_IMPL_HPP

// includes
#include "pitts_machine_info.hpp"
#include "pitts_parallel.hpp"
#ifdef PITTS_HAVE_LIKWID
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
  namespace machine_info
  {
    //! cached results with machine information
    static inline MachineInfo cachedMachineInfo{};
  }


  // implement initialize
  INLINE MachineInfo getMachineInfo(bool initialize)
  {
    if( !initialize )
      return machine_info::cachedMachineInfo;

    // query actual machine information, using internal Eigen interface for now
    MachineInfo mi;
#ifdef PITTS_HAVE_LIKWID
    topology_init();
    const auto cpuTopo = get_cpuTopology();

    const auto total_cores = cpuTopo->numHWThreads / cpuTopo->numThreadsPerCore;
    // just assuming each thread uses one core -> only a heuristic!
    int openmpThreads = 0;
#pragma omp parallel
    {
      const auto& [iThread, nThreads] = internal::parallel::ompThreadInfo();
      if( iThread == 0 )
        openmpThreads = nThreads;
    }
    const int available_cores = std::min<int>(total_cores, openmpThreads);

    for(int i = 0; i < cpuTopo->numCacheLevels; i++)
    {
      const auto cache = cpuTopo->cacheLevels[i];
      if( cache.type != DATACACHE && cache.type != UNIFIEDCACHE )
        continue;

      auto cache_cores = cache.threads / cpuTopo->numThreadsPerCore;

      if( cache.level == 1 )
        mi.cacheSize_L1_perCore = cache.size / cache_cores;
      if( cache.level == 2 )
        mi.cacheSize_L2_perCore = cache.size / cache_cores;
      if( cache.level == 3 )
        mi.cacheSize_L3_total = cache.size / cache_cores * available_cores;
    }
    topology_finalize();
#endif

    machine_info::cachedMachineInfo = mi;

    return mi;
  }

}


#endif // PITTS_MACHINE_INFO_IMPL_HPP
