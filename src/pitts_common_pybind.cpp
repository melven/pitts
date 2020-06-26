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
#include "pitts_common.hpp"

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
          [](bool verbose){PITTS::initialize(nullptr, nullptr, verbose);},
          py::arg("verbose")=true,
          "Call MPI_Init if needed and print some general information");
      m.def("finalize",
          &PITTS::finalize,
          py::arg("verbose")=true,
          "Call MPI_Finalize if needed and print some statistics");
    }
  }
}
