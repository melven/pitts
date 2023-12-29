// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_common_jlbind.cpp
* @brief Julia binding for PITTS common helper functions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-12-18
*
**/

// includes
#include <jlcxx/jlcxx.hpp>
#include "pitts_common_jlbind.hpp"
#include "pitts_mkl.hpp"
#include "pitts_common.hpp"
#include "pitts_performance.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for Julia bindings
  namespace jlbind
  {
    // create jlcxx-wrapper for PITTS::initialize and PITTS::finalize
    void define_common(jlcxx::Module& m)
    {
      m.method("initialize",
          [](bool verbose = true, std::uint_fast64_t randomSeed = internal::generateRandomSeed()){PITTS::initialize(nullptr, nullptr, verbose, randomSeed);});
      m.method("finalize", PITTS::finalize);
      m.method("printPerformanceStatistics", 
          [](bool clear = true, bool mpiGlobal = true){PITTS::performance::printStatistics(clear, std::cout, mpiGlobal);});
      m.method("clearPerformanceStatistics", &PITTS::performance::clearStatistics);
    }
  }
}
