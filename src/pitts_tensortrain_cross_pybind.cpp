// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_cross_pybind.cpp
* @brief python binding for the TT-cross approximation (element-wise functions)
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-12
*
**/

// includes
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "pitts_multivector.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_cross.hpp"
#include "pitts_tensortrain_cross_pybind.hpp"

namespace py = pybind11;


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for python bindings
  namespace pybind
  {
    namespace
    {
      //! call a python function with all entries of a local block of the TT-cross algorithm
      //!
      //! The block is passed as a 1d numpy array that just views the data of the MultiVector (no copy),
      //! so it is only valid during the call. The python function either modifies it in-place and returns
      //! None, or returns an array of the same size (e.g. a numpy ufunc like numpy.sqrt).
      //!
      template<typename T>
      void applyPythonFunction(const py::object& f, MultiVector<T>& mv)
      {
        // the surrounding applyElementwise call released the GIL
        const py::gil_scoped_acquire acquireGil;

        const py::array_t<T> entries(py::array::ShapeContainer{mv.rows()},
                                     py::array::StridesContainer{static_cast<py::ssize_t>(sizeof(T))},
                                     &mv(0, 0), py::capsule(&mv, [](void*){}));

        const py::object result = f(entries);
        if( result.is_none() )
          return;

        const auto resultEntries = result.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
        if( resultEntries.size() != mv.rows() )
          throw std::invalid_argument("applyElementwise: the function must return an array with the same number of entries!");
        std::copy_n(resultEntries.data(), mv.rows(), &mv(0, 0));
      }

      //! wrapper for PITTS::applyElementwise with the options as individual arguments
      template<typename T>
      void TensorTrain_applyElementwise(const py::object& f, const TensorTrain<T>& TTx, TensorTrain<T>& TTy,
                                        T rankTolerance, int maxRank, int nSweeps, T convergenceTolerance,
                                        int nMALS, int errorRank, int enrichmentRank, bool randomErrorIndexSets,
                                        T maxvolTolerance, bool verbose)
      {
        TTCrossOptions<T> options;
        options.rankTolerance = rankTolerance;
        options.maxRank = maxRank;
        options.nSweeps = nSweeps;
        options.convergenceTolerance = convergenceTolerance;
        options.nMALS = nMALS;
        options.errorRank = errorRank;
        options.enrichmentRank = enrichmentRank;
        options.randomErrorIndexSets = randomErrorIndexSets;
        options.maxvolTolerance = maxvolTolerance;
        options.verbose = verbose;

        // release the GIL for the (possibly OpenMP/MPI parallel) parts of the algorithm,
        // applyPythonFunction acquires it again for each block
        const py::gil_scoped_release releaseGil;

        applyElementwise([&f](MultiVector<T>& mv){applyPythonFunction<T>(f, mv);}, TTx, TTy, options);
      }

      //! provide the TT-cross functions for one data type
      template<typename T>
      void init_TensorTrain_cross_helper(py::module& m)
      {
        const TTCrossOptions<T> defaults;

        m.def("applyElementwise",
            &TensorTrain_applyElementwise<T>,
            py::arg("f"), py::arg("TTx"), py::arg("TTy"),
            py::arg("rankTolerance") = defaults.rankTolerance, py::arg("maxRank") = defaults.maxRank,
            py::arg("nSweeps") = defaults.nSweeps, py::arg("convergenceTolerance") = defaults.convergenceTolerance,
            py::arg("nMALS") = defaults.nMALS, py::arg("errorRank") = defaults.errorRank,
            py::arg("enrichmentRank") = defaults.enrichmentRank,
            py::arg("randomErrorIndexSets") = defaults.randomErrorIndexSets,
            py::arg("maxvolTolerance") = defaults.maxvolTolerance, py::arg("verbose") = defaults.verbose,
            "Calculate TTy ~ f(TTx) element-wise using the TT-cross approximation\n\n"
            "f is called with a 1d numpy array of tensor entries (a small part of the tensor) and must return\n"
            "an array of the same size, or None if it modified its argument in-place. So f must be a\n"
            "vectorized function (e.g. numpy.sqrt or numpy.vectorize(myScalarFunction)); it is called several\n"
            "times per sweep and dimension. The array only views internal data: do not keep a reference to it.");
      }
    }

    // create pybind11-wrapper for PITTS::applyElementwise (TT-cross)
    void init_TensorTrain_cross(py::module& m)
    {
      init_TensorTrain_cross_helper<float>(m);
      init_TensorTrain_cross_helper<double>(m);
    }
  }
}
