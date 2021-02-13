/*! @file pitts_tensortrain_operator_pybind.hpp
* @brief header for python binding for PITTS::TensorTrainOperator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2021-02-12
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/


// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_PYBIND_HPP
#define PITTS_TENSORTRAIN_OPERATOR_PYBIND_HPP

// includes
#include <pybind11/pybind11.h>


namespace py = pybind11;


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for python bindings
  namespace pybind
  {
    //! create pybind11-wrapper for PITTS::TensorTrainOperator
    void init_TensorTrainOperator(py::module& m);
  }
}

#endif // PITTS_TENSORTRAIN_OPERATOR_PYBIND_HPP
