// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_cross_pybind.cpp
* @brief python binding for the TT-cross approximation (element-wise functions, also of symmetric tensors)
* @author Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2026-09-12
*
**/

// includes
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
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
      //! For a function of several tensors, the blocks of the further inputs are passed as further arguments
      //! (also views, they must not be modified).
      //!
      template<typename T, typename... Others>
      void applyPythonFunction(const py::object& f, MultiVector<T>& mv, const Others&... others)
      {
        // the surrounding applyElementwise call released the GIL
        const py::gil_scoped_acquire acquireGil;

        const auto view = [](const MultiVector<T>& block)
        {
          return py::array_t<T>(py::array::ShapeContainer{block.rows()},
                                py::array::StridesContainer{static_cast<py::ssize_t>(sizeof(T))},
                                &block(0, 0), py::capsule(&block, [](void*){}));
        };

        const py::object result = f(view(mv), view(others)...);
        if( result.is_none() )
          return;

        const auto resultEntries = result.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
        if( resultEntries.size() != mv.rows() )
          throw std::invalid_argument("applyElementwise: the function must return an array with the same number of entries!");
        std::copy_n(resultEntries.data(), mv.rows(), &mv(0, 0));
      }

      //! wrapper for PITTS::applyElementwise (or applyElementwiseSymmetric) with one or more input tensors (Inputs = TensorTrain<T>...)
      //! and the options as individual arguments
      template<bool symmetric, typename T, typename... Inputs>
      void TensorTrain_applyElementwise(const py::object& f, const Inputs&... TTx, TensorTrain<T>& TTy,
                                        T rankTolerance, int maxRank, int nSweeps, T convergenceTolerance,
                                        int nMALS, int errorRank, int enrichmentRank, bool randomErrorIndexSets,
                                        T maxvolTolerance, T oversampling, int controlSampleSize, bool verbose)
      {
        // release the GIL for the (possibly OpenMP/MPI parallel) parts of the algorithm,
        // applyPythonFunction acquires it again for each block
        const py::gil_scoped_release releaseGil;

        // array function with one block per input (the result in the first one)
        const auto pythonFunction = [&f](MultiVector<T>& mv, const auto&... others)
        {
          applyPythonFunction<T>(f, mv, others...);
        };
        const TTCrossOptions<T> options{.rankTolerance = rankTolerance, .maxRank = maxRank, .nSweeps = nSweeps,
                                        .convergenceTolerance = convergenceTolerance, .nMALS = nMALS,
                                        .errorRank = errorRank, .enrichmentRank = enrichmentRank,
                                        .randomErrorIndexSets = randomErrorIndexSets,
                                        .maxvolTolerance = maxvolTolerance, .oversampling = oversampling,
                                        .controlSampleSize = controlSampleSize, .verbose = verbose};
        if constexpr( symmetric )
          applyElementwiseSymmetric(pythonFunction, TTx..., TTy, options);
        else
          applyElementwise(pythonFunction, TTx..., TTy, options);
      }

      //! define applyElementwise or applyElementwiseSymmetric for one data type and the given input tensors (Inputs = TensorTrain<T>..., with their argument names)
      template<bool symmetric, typename T, typename... Inputs>
      void defineApplyElementwise(py::module& m, const std::array<const char*, sizeof...(Inputs)>& inputNames, const char* doc)
      {
        const TTCrossOptions<T> defaults;

        [&]<std::size_t... i>(std::index_sequence<i...>)
        {
          m.def(symmetric ? "applyElementwiseSymmetric" : "applyElementwise",
            &TensorTrain_applyElementwise<symmetric, T, Inputs...>,
            py::arg("f"), py::arg(inputNames[i])..., py::arg("TTy"),
            py::arg("rankTolerance") = defaults.rankTolerance, py::arg("maxRank") = defaults.maxRank,
            py::arg("nSweeps") = defaults.nSweeps, py::arg("convergenceTolerance") = defaults.convergenceTolerance,
            py::arg("nMALS") = defaults.nMALS, py::arg("errorRank") = defaults.errorRank,
            py::arg("enrichmentRank") = defaults.enrichmentRank,
            py::arg("randomErrorIndexSets") = defaults.randomErrorIndexSets,
            py::arg("maxvolTolerance") = defaults.maxvolTolerance, py::arg("oversampling") = defaults.oversampling,
            py::arg("controlSampleSize") = defaults.controlSampleSize, py::arg("verbose") = defaults.verbose,
            doc);
        }(std::index_sequence_for<Inputs...>{});
      }

      //! provide the TT-cross functions for one data type
      template<typename T>
      void init_TensorTrain_cross_helper(py::module& m)
      {
        defineApplyElementwise<false, T, TensorTrain<T>>(m, {"TTx"},
            "Calculate TTy ~ f(TTx) element-wise using the TT-cross approximation\n\n"
            "f is called with a 1d numpy array of tensor entries (a small part of the tensor) and must return\n"
            "an array of the same size, or None if it modified its argument in-place. So f must be a\n"
            "vectorized function (e.g. numpy.sqrt or numpy.vectorize(myScalarFunction)); it is called several\n"
            "times per sweep and dimension. The array only views internal data: do not keep a reference to it.");

        defineApplyElementwise<false, T, TensorTrain<T>, TensorTrain<T>>(m, {"TTx1", "TTx2"},
            "Calculate TTy ~ f(TTx1, TTx2) element-wise using the TT-cross approximation, e.g. the element-wise product\n\n"
            "f is called with two 1d numpy arrays of corresponding entries of TTx1 and TTx2 and must return an\n"
            "array of the same size (e.g. numpy.multiply), or None if it stored the result in its first argument.\n"
            "It must not modify the second argument. Otherwise as applyElementwise(f, TTx, TTy).");

        defineApplyElementwise<true, T, TensorTrain<T>>(m, {"TTx"},
            "Calculate y ~ f(x) element-wise for a symmetric x = TTx TTx^T and a positive semidefinite result y = TTy TTy^T\n\n"
            "TTx and TTy are left halves: tensor trains with half of the dimensions whose last sub-tensor has an open right\n"
            "rank (set it with setSubTensor), x(i_0..i_{2m-1}) = TTx(i_0..i_{m-1},:) . TTx(i_{2m-1}..i_m,:), e.g. a Gram matrix.\n"
            "f must be positive semidefinite on x (e.g. numpy.exp of a Gram matrix); negative eigenvalues are removed.\n"
            "f is called as in applyElementwise(f, TTx, TTy); the same options (randomErrorIndexSets is not supported).");

        defineApplyElementwise<true, T, TensorTrain<T>, TensorTrain<T>>(m, {"TTx1", "TTx2"},
            "Calculate y ~ f(x1, x2) element-wise for symmetric x1 = TTx1 TTx1^T, x2 = TTx2 TTx2^T, y = TTy TTy^T (e.g. numpy.multiply)\n\n"
            "f is called as in applyElementwise(f, TTx1, TTx2, TTy); otherwise as applyElementwiseSymmetric(f, TTx, TTy).");
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
