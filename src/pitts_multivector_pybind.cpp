/*! @file pitts_multivector_pybind.cpp
* @brief python binding for PITTS::MultiVector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-07-16
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// includes
#include <iostream>
#include <variant>
#include <string>
#include <exception>
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_cdist.hpp"
#include "pitts_multivector_centroids.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_transpose.hpp"
#include "pitts_multivector_pybind.hpp"
#include "pitts_tensor2.hpp"
#include "pitts_scope_info.hpp"

// include pybind11 last (workaround for problem with C++20 modules)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <pybind11/complex.h>
#include <pybind11/numpy.h>

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
      //! helper function to copy PITTS::MultiVector into a numpy array
      template<typename T>
      py::array_t<T> copy(const Tensor2<T>& buff)
      {
        py::array_t<T> result({buff.r1(), buff.r2()});
        for(long long i = 0; i < buff.r1(); i++)
          for(long long j = 0; j < buff.r2(); j++)
            *result.mutable_data(i,j) = buff(i,j);
        return result;
      }

      //! helper function to print the attributes of the MultiVector object nicely
      template<typename T>
      std::string MultiVector_toString(const MultiVector<T>& mv)
      {
        // helper for getting the template type nicely formatted
        constexpr auto scope = internal::ScopeInfo::current<T>();

        return "PITTS::MultiVector" + std::string(scope.type_name()) + "(" + std::to_string(mv.rows()) + ", " + std::to_string(mv.cols()) + ")";
      }

      //! provide all MultiVector<T> related classes and functions
      template<typename T>
      void init_MultiVector_helper(py::module& m, const std::string& type_name)
      {
        const std::string className = "MultiVector_" + type_name;

        py::class_<MultiVector<T>>(m, className.c_str(), py::buffer_protocol(), "Simple multi-vector class")
          .def(py::init<long long,long long>(), py::arg("rows")=0, py::arg("cols")=0, "Create MultiVector with given dimensions")
          .def_buffer([](MultiVector<T>& mv) {
              return py::buffer_info(&mv(0,0), sizeof(T), py::format_descriptor<T>::format(), 2, {mv.rows(), mv.cols()}, {sizeof(T), sizeof(T)*(&mv(0,1)-&mv(0,0))});
              })
          .def("rows", &MultiVector<T>::rows, "number of rows")
          .def("cols", &MultiVector<T>::cols, "number of columns")
          .def("resize", &MultiVector<T>::resize, "change the number of rows and columns (destroying all data!)")
          .def("__str__", &MultiVector_toString<T>, "Print the attributes of the given MultiVector object");

        m.def("copy",
            py::overload_cast< const MultiVector<T>&, MultiVector<T>& >(&PITTS::copy<T>),
            py::arg("source"), py::arg("destination"),
            "explicitly copy a MultiVector object");

        m.def("randomize",
            py::overload_cast< MultiVector<T>& >(&randomize<T>),
            py::arg("mv"),
            "fill a multi-vector with random values (keeping current dimensions)");

        m.def("centroids",
            py::overload_cast< const MultiVector<T>&, const std::vector<long long>&, const std::vector<T>&, MultiVector<T>& >(&centroids<T>),
            py::arg("X"), py::arg("idx"), py::arg("weights"), py::arg("Y"),
            "Calculate weighted sum of columns of X and store the result in Y: (sums up Y with Y(idx_i) += weights_i*X_i)");

        m.def("cdist2",
            [](const MultiVector<T>& X, const MultiVector<T>& Y) {
              Tensor2<T> buff;
              cdist2(X, Y, buff);
              return copy(buff);
              },
            py::arg("X"), py::arg("Y"),
            "Calculate the squared distance of each vector in one multi-vector X with each vector in another (small) multi-vector Y");

        m.def("block_TSQR",
            [](const MultiVector<T>& M, int reductionFactor, bool mpiGlobal) {
              Tensor2<T> buff;
              block_TSQR(M, buff, reductionFactor, mpiGlobal);
              return copy(buff);
              },
            py::arg("M"), py::arg("reductionFactor")=0, py::arg("mpiGlobal")=true,
            "Calculate upper triangular part R from a QR-decomposition of the given tall-skinny matrix (multi-vector) M");

        m.def("transform",
            py::overload_cast< const MultiVector<T>&, const Tensor2<T>&, MultiVector<T>&, std::array<long long,2> >(&PITTS::transform<T>),
            py::arg("X"), py::arg("M"), py::arg("Y"), py::arg("reshape")=std::array<long long,2>{0,0},
            "Calculate the matrix-matrix product of a tall-skinny matrix (multivector) with a small matrix (Y <- X*M)");

        m.def("transpose",
            py::overload_cast< const MultiVector<T>&, MultiVector<T>&, std::array<long long,2>, bool >(&PITTS::transpose<T>),
            py::arg("X"), py::arg("Y"), py::arg("reshape")=std::array<long long,2>{0,0}, py::arg("reverse")=false,
            "Reshape and transpose a tall-skinny matrix");
      }
    }

    // create pybind11-wrapper for PITTS::TensorTrain
    void init_MultiVector(py::module& m)
    {
      init_MultiVector_helper<float>(m, "float");
      init_MultiVector_helper<double>(m, "double");
      //init_MultiVector_helper<std::complex<float>>(m, "float_complex");
      //init_MultiVector_helper<std::complex<double>>(m, "double_complex");
    }
  }
}
