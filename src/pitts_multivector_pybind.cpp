// Copyright (c) 2020 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_multivector_pybind.cpp
* @brief python binding for PITTS::MultiVector
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-07-16
*
**/

// includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <string>
#include <stdexcept>
#include "pitts_multivector.hpp"
#include "pitts_multivector_random.hpp"
#include "pitts_multivector_cdist.hpp"
#include "pitts_multivector_centroids.hpp"
#include "pitts_multivector_tsqr.hpp"
#include "pitts_multivector_transform.hpp"
#include "pitts_multivector_transpose.hpp"
#include "pitts_multivector_pybind.hpp"
#include "pitts_ttn_local_gradient_contraction.hpp"
#include "pitts_scope_info.hpp"

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

      //! helper function that wraps ttn_local_gradient_contract
      template<typename T>
      py::array_t<T> wrap_ttn_local_gradient_contract(py::array_t<T> optTensor, const MultiVector<T>& envLeft, const MultiVector<T>& envRight, const MultiVector<T>& envTop, bool mpiParallel)
      {
        if( optTensor.ndim() != 3 )
          throw std::invalid_argument("array must have 3 dimensions");

        Tensor3<T> optTensor_t3(optTensor.shape(0), optTensor.shape(1), optTensor.shape(2));
        for(long long i2 = 0; i2 < optTensor_t3.r2(); i2++)
          for(long long j = 0; j < optTensor_t3.n(); j++)
            for(long long i1 = 0; i1 < optTensor_t3.r1(); i1++)
              optTensor_t3(i1,j,i2) = *optTensor.data(i1,j,i2);

        Tensor3<T> gradOptTensor_t3(optTensor.shape(0), optTensor.shape(1), optTensor.shape(2));
        PITTS::ttn_local_gradient_contract(optTensor_t3, envLeft, envRight, envTop, gradOptTensor_t3, mpiParallel);

        py::array_t<T> result({gradOptTensor_t3.r1(), gradOptTensor_t3.n(), gradOptTensor_t3.r2()});
        for(long long i2 = 0; i2 < gradOptTensor_t3.r2(); i2++)
          for(long long j = 0; j < gradOptTensor_t3.n(); j++)
            for(long long i1 = 0; i1 < gradOptTensor_t3.r1(); i1++)
              *result.mutable_data(i1,j,i2) = gradOptTensor_t3(i1,j,i2);

        return result;
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
          .def("resize", &MultiVector<T>::resize, py::arg("rows"), py::arg("cols"), py::arg("setPaddingToZero")=true, py::arg("keepData")=false, "change the number of rows and columns (destroying all data!)")
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
            py::overload_cast< const MultiVector<T>&, const ConstTensor2View<T>&, MultiVector<T>&, std::array<long long,2> >(&PITTS::transform<T>),
            py::arg("X"), py::arg("M"), py::arg("Y"), py::arg("reshape")=std::array<long long,2>{0,0},
            "Calculate the matrix-matrix product of a tall-skinny matrix (multivector) with a small matrix (Y <- X*M)");

        m.def("transpose",
            py::overload_cast< const MultiVector<T>&, MultiVector<T>&, std::array<long long,2>, bool >(&PITTS::transpose<T>),
            py::arg("X"), py::arg("Y"), py::arg("reshape")=std::array<long long,2>{0,0}, py::arg("reverse")=false,
            "Reshape and transpose a tall-skinny matrix");

        m.def("ttn_local_gradient_contraction",
            py::overload_cast<py::array_t<T>, const MultiVector<T>&, const MultiVector<T>&,  const MultiVector<T>&, bool>(&wrap_ttn_local_gradient_contract<T>),
            py::arg("optTensor"), py::arg("envLeft"), py::arg("envRight"), py::arg("envTop"), py::arg("mpiParallel")=false,
            "Tree tensor network (TTN) local gradient contraction");
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
