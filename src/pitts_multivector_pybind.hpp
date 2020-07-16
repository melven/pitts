/*! @file pitts_multivector_pybind.hpp
* @brief header for python binding for PITTS::MultiVector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/


// include guard
#ifndef PITTS_MULTIVECTOR_PYBIND_HPP
#define PITTS_MULTIVECTOR_PYBIND_HPP

// includes
#include <pybind11/pybind11.h>


namespace py = pybind11;


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for python bindings
  namespace pybind
  {
    //! create pybind11-wrapper for PITTS::MultiVector
    void init_MultiVector(py::module& m);
  }
}

#endif // PITTS_MULTIVECTOR_PYBIND_HPP
