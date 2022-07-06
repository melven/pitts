/*! @file pitts_tensortrain_solve_pybind.hpp
* @brief header for python binding for high-level solvers for linear system in tensor-train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-07-06
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/


// include guard
#ifndef PITTS_TENSORTRAIN_SOLVE_PYBIND_HPP
#define PITTS_TENSORTRAIN_SOLVE_PYBIND_HPP

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
    void init_TensorTrain_solve(py::module& m);
  }
}

#endif // PITTS_TENSORTRAIN_SOLVE_PYBIND_HPP
