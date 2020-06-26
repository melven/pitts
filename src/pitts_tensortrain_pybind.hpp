/*! @file pitts_tensortrain_pybind.hpp
* @brief header for python binding for PITTS::TensorTrain
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/


// include guard
#ifndef PITTS_TENSORTRAIN_PYBIND_HPP
#define PITTS_TENSORTRAIN_PYBIND_HPP

// includes
#include <pybind11/pybind11.h>


namespace py = pybind11;


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for python bindings
  namespace pybind
  {
    //! create pybind11-wrapper for PITTS::TensorTrain
    void init_TensorTrain(py::module& m);
  }
}

#endif // PITTS_TENSORTRAIN_PYBIND_HPP
