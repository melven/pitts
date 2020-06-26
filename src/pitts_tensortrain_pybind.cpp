/*! @file pitts_tensortrain_pybind.cpp
* @brief python binding for PITTS::TensorTrain
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// includes
#include <pybind11/stl.h>
//#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <string>
#include <string_view>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_from_dense.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_tensortrain_pybind.hpp"

namespace py = pybind11;


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for python bindings
  namespace pybind
  {
    //! internal namespace for helper functions
    namespace
    {
      //! helper function for fromDense -> TensorTrain
      template<typename T>
      TensorTrain<T> TensorTrain_fromDense(py::array_t<T, py::array::f_style> array, T rankTolerance)
      {
        const auto first = static_cast<const T*>(array.data());
        const auto last = first + array.size();
        auto shape = std::vector<int>{array.shape(), array.shape()+array.ndim()};
        return fromDense(first, last, shape, rankTolerance);
      }

      //! helper function for TensorTrain -> toDense
      template<typename T>
      py::array_t<T, py::array::f_style> TensorTrain_toDense(const TensorTrain<T>& TT)
      {
        py::array_t<T, py::array::f_style> array(TT.dimensions());
        auto first = static_cast<T*>(array.mutable_data());
        auto last = first + array.size();

        toDense(TT, first, last);

        return array;
      }

      //! provide all TensorTrain<T> related classes and functions
      template<typename T>
      void init_TensorTrain_helper(py::module& m, const std::string& type_name)
      {
        const std::string className = "TensorTrain_" + type_name;

        py::class_<TensorTrain<T>>(m, className.c_str(), "Simple tensor train class")
          .def(py::init< const std::vector<int>& >(), py::arg("dimensions"), "Create TensorTrain with given dimensions")
          .def("dimensions", &TensorTrain<T>::dimensions, "tensor dimensions (immutable)")
          .def("setTTranks", py::overload_cast<int>(&TensorTrain<T>::setTTranks), py::arg("tt_rank"), "set sub-tensor dimensions (TT-ranks), destroying all existing data")
          .def("setTTranks", py::overload_cast<const std::vector<int>& >(&TensorTrain<T>::setTTranks), py::arg("tt_ranks"), "set sub-tensor dimensions (TT-ranks), destroying all existing data")
          .def("getTTranks", &TensorTrain<T>::getTTranks, "get current sub-tensor dimensions (TT-ranks)")
          .def("setZero", &TensorTrain<T>::setZero, "make this a tensor of zeros")
          .def("setOnes", &TensorTrain<T>::setOnes, "make this a tensor of ones")
          .def("setUnit", &TensorTrain<T>::setUnit, py::arg("index"), "make this a canonical unit tensor in the given direction");

        m.def("copy",
            py::overload_cast< const TensorTrain<T>&, TensorTrain<T>& >(&copy<T>),
            py::arg("source"), py::arg("destination"),
            "explicitly copy a TensorTrain object");

        m.def("axpby",
            py::overload_cast< T, const TensorTrain<T>&, T, TensorTrain<T>&, T>(&axpby<T>),
            py::arg("alpha"), py::arg("TTx"), py::arg("beta"), py::arg("TTy"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()),
            "Scale and add one tensor train to another\n\nCalculate gamma*TTy <- alpha*TTx + beta*TTy\n\nBoth tensors must be leftNormalized, gamma is returned");

        m.def("dot",
            py::overload_cast< const TensorTrain<T>&, const TensorTrain<T>& >(&dot<T>),
            py::arg("TTx"), py::arg("TTy"),
            "calculate the inner product of two vectors in tensor train format");

        m.def("norm2",
            py::overload_cast< const TensorTrain<T>& >(&norm2<T>),
            py::arg("TT"),
            "calculate the 2-norm for a vector in tensor train format");

        m.def("normalize",
            py::overload_cast< TensorTrain<T>&, T >(&normalize<T>),
            py::arg("TT"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()),
            "TT-rounting: truncate tensor train by two normalization sweeps (first right to left, then left to right)");

        m.def("leftNormalize",
            py::overload_cast< TensorTrain<T>&, T >(&leftNormalize<T>),
            py::arg("TT"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()),
            "Make all sub-tensors orthogonal sweeping left to right");

        m.def("rightNormalize",
            py::overload_cast< TensorTrain<T>&, T >(&rightNormalize<T>),
            py::arg("TT"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()),
            "Make all sub-tensors orthogonal sweeping right to left");

        m.def("randomize",
            py::overload_cast< TensorTrain<T>& >(&randomize<T>),
            py::arg("TT"),
            "fill a tensor train format with random values (keeping current TT-ranks)");

        m.def("fromDense",
            &TensorTrain_fromDense<T>,
            py::arg("array"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()),
            "calculate tensor-train decomposition of a tensor stored in fully dense format");

        m.def("toDense",
            &TensorTrain_toDense<T>,
            py::arg("TT"),
            "calculate fully dense tensor from a tensor-train decomposition");
      }
    }

    // create pybind11-wrapper for PITTS::TensorTrain
    void init_TensorTrain(py::module& m)
    {
      init_TensorTrain_helper<float>(m, "float");
      init_TensorTrain_helper<double>(m, "double");
      //init_TensorTrain_helper<std::complex<float>>(m, "float_complex");
      //init_TensorTrain_helper<std::complex<double>>(m, "double_complex");
    }
  }
}
