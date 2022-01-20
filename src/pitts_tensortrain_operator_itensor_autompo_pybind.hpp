/*! @file pitts_tensortrain_operator_itensor_autompo_pybind.hpp
* @brief header for python binding of the PITTS wrapper for ITensor autompo
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-01-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/


// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_ITENSOR_AUTOMPO_PYBIND_HPP
#define PITTS_TENSORTRAIN_OPERATOR_ITENSOR_AUTOMPO_PYBIND_HPP

// includes
#include <pybind11/pybind11.h>


namespace py = pybind11;


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for python bindings
  namespace pybind
  {
    //! create pybind11-wrapper for ITensor autompo functionality to create PITT::TensorTrainOperator
    void init_TensorTrainOperator_itensor_autompo(py::module& m);
  }
}

#endif // PITTS_TENSORTRAIN_OPERATOR_ITENSOR_AUTOMPO_PYBIND_HPP
