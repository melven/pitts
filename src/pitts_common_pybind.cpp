/*! @file pitts_common_pybind.cpp
* @brief python binding for PITTS common helper functions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
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
          "Call MPI_Finalize if needed and print some statistics");
      m.def("printPerformanceStatistics",
          [](bool clear, bool mpiGlobal){PITTS::performance::printStatistics(clear, std::cout, mpiGlobal);},
          //py::arg("clear")=true,// py::arg("out")=std::cout, py::arg("mpiGlobal")=true, // std::cout not supported as default argument?
          py::arg("clear")=true, py::arg("mpiGlobal")=true,
          "Print nice statistics using globalPerformanceStatisticsMap");
      m.def("clearPerformanceStatistics",
          &PITTS::performance::clearStatistics,
          "Clear globalPerformanceStatisticsMap");
    }
  }
}
