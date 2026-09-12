// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_cross_pybind.hpp
* @brief header for python binding for the TT-cross approximation (element-wise functions)
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-12
*
**/


// include guard
#ifndef PITTS_TENSORTRAIN_CROSS_PYBIND_HPP
#define PITTS_TENSORTRAIN_CROSS_PYBIND_HPP

// includes
#include <pybind11/pybind11.h>


namespace py = pybind11;


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for python bindings
  namespace pybind
  {
    //! create pybind11-wrapper for PITTS::applyElementwise (TT-cross)
    void init_TensorTrain_cross(py::module& m);
  }
}

#endif // PITTS_TENSORTRAIN_CROSS_PYBIND_HPP
