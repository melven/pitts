// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_common_pybind.cpp
* @brief python binding for PITTS common helper functions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-26
*
**/

// includes
#include <pybind11/pybind11.h>
#include "pitts_common_pybind.hpp"
#include "pitts_mkl.hpp"
#include "pitts_common.hpp"
#include "pitts_performance.hpp"

namespace py = pybind11;


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for python bindings
  namespace pybind
  {
    // create pybind11-wrapper for PITTS::initialize and PITTS::finalize
    void init_common(py::module& m)
    {
      m.def("initialize",
          [](bool verbose, std::uint_fast64_t randomSeed){PITTS::initialize(nullptr, nullptr, verbose, randomSeed);},
          py::arg("verbose")=true, py::arg("randomSeed")=internal::generateRandomSeed(),
          "Call MPI_Init if needed and print some general information");
      m.def("finalize",
          &PITTS::finalize,
          py::arg("verbose")=true,
          // the statistics can be large, see printPerformanceStatistics
          py::call_guard<py::gil_scoped_release>(),
          "Call MPI_Finalize if needed and print some statistics");
      m.def("printPerformanceStatistics",
          [](bool clear, bool mpiGlobal){PITTS::performance::printStatistics(clear, std::cout, mpiGlobal);},
          //py::arg("clear")=true,// py::arg("out")=std::cout, py::arg("mpiGlobal")=true, // std::cout not supported as default argument?
          py::arg("clear")=true, py::arg("mpiGlobal")=true,
          // Release the GIL while writing to std::cout: jupyter captures the raw file descriptors and
          // drains the pipe in a python thread, which needs the GIL. Writing more than the pipe buffer
          // (64 KiB, easily reached by these statistics) while holding the GIL deadlocks the kernel.
          py::call_guard<py::gil_scoped_release>(),
          "Print nice statistics using globalPerformanceStatisticsMap");
      m.def("clearPerformanceStatistics",
          &PITTS::performance::clearStatistics,
          "Clear globalPerformanceStatisticsMap");
    }
  }
}
