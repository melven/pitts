// Copyright (c) 2024 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_machine_info.hpp
* @brief Helper functionality for machine-specific parameters like cache sizes
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2024-07-15
*
**/

// include guard
#ifndef PITTS_MACHINE_INFO_HPP
#define PITTS_MACHINE_INFO_HPP

// includes

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Helper struct with useful information on the machine this code currently runs on
  struct MachineInfo
  {
    //! L1 cache size in bytes per core, -1 if not available
    int cacheSize_L1_perCore = -1;

    //! L2 cache size in bytes per core, -1 if not available
    int cacheSize_L2_perCore = -1;

    //! L3 cache size in bytes currently available (total/#cores*#threads), -1 if not available
    int cacheSize_L3_total = -1;
  };

  //! get information on the machine (cached)
  //!
  //! @warning Only available after PITTS::initialize was called!
  //!
  //! @param initialize (internal) Actually query machine data, only used from PITTS::initialize and for testing.
  //! @return machine data
  //!
  MachineInfo getMachineInfo(bool initialize = false);
}


#ifndef PITTS_DEVELOP_BUILD
#include "pitts_machine_info_impl.hpp"
#endif

#endif // PITTS_MACHINE_INFO_HPP
