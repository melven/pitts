/*! @file pitts_tensortrain_pybind.cpp
* @brief python binding for PITTS::TensorTrain
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <string>
#include <exception>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_dot.hpp"
#include "pitts_tensortrain_norm.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_from_dense_classical.hpp"
#include "pitts_tensortrain_from_dense.hpp"
#include "pitts_tensortrain_from_dense_twosided.hpp"
#include "pitts_tensortrain_to_dense.hpp"
#include "pitts_tensortrain_gram_schmidt.hpp"
#include "pitts_tensortrain_pybind.hpp"
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
      //! helper function for converting an array to a string
      template<typename T>
      std::string to_string(const std::vector<T>& v)
      {
        std::string result = "[";
        for(int i = 0; i < v.size(); i++)
        {
          if( i > 0 )
            result += ", ";
          result += std::to_string(v[i]);
        }
        result += "]";
        return result;
      }

      //! helper function to obtain a sub-tensor from a TensorTrain
      template<typename T>
      py::array_t<T> TensorTrain_getSubTensor(const TensorTrain<T>& TT, int d)
      {
        const auto& subT = TT.subTensor(d);
        py::array_t<T> array({subT.r1(), subT.n(), subT.r2()});

        for(int i2 = 0; i2 < subT.r2(); i2++)
          for(int j = 0; j < subT.n(); j++)
            for(int i1 = 0; i1 < subT.r1(); i1++)
              *array.mutable_data(i1,j,i2) = subT(i1,j,i2);
        return array;
      }

      //! helper function to set a sub-tensor in a TensorTrain
      template<typename T>
      void TensorTrain_setSubTensor(TensorTrain<T>& TT, int d, py::array_t<T> array)
      {
        PITTS::Tensor3<T> subT(TT.subTensor(d).r1(), TT.subTensor(d).n(), TT.subTensor(d).r2());
        if( array.ndim() != 3 )
          throw std::invalid_argument("array must have 3 dimensions");
        if( array.shape(0) != subT.r1() || array.shape(1) != subT.n() || array.shape(2) != subT.r2() )
          throw std::invalid_argument("array has incompatible shape, expected " + to_string<int>({subT.r1(), subT.n(), subT.r2()}) + ", got " + to_string<int>({array.shape(), array.shape()+array.ndim()}));

        for(int i2 = 0; i2 < subT.r2(); i2++)
          for(int j = 0; j < subT.n(); j++)
            for(int i1 = 0; i1 < subT.r1(); i1++)
              subT(i1,j,i2) = *array.data(i1,j,i2);

        TT.setSubTensor(d, std::move(subT));
      }

      //! helper function for fromDense -> TensorTrain
      template<typename T>
      TensorTrain<T> TensorTrain_fromDense_classical(py::array_t<T, py::array::f_style> array, T rankTolerance, int maxRank)
      {
        const auto first = static_cast<const T*>(array.data());
        const auto last = first + array.size();
        auto shape = std::vector<int>{array.shape(), array.shape()+array.ndim()};
        return fromDense_classical(first, last, shape, rankTolerance, maxRank);
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

      //! helper function to print the attributes of the TensorTrain object nicely
      template<typename T>
      std::string TensorTrain_toString(const TensorTrain<T>& TT)
      {
        // helper for getting the template type nicely formatted
        constexpr auto scope = internal::ScopeInfo::current<T>();

        std::string result;
        result += "PITTS::TensorTrain" + std::string(scope.type_name()) + "\n";
        result += "        with dimensions = " + to_string(TT.dimensions()) + "\n";
        result += "             ranks      = " + to_string(TT.getTTranks());

        return result;
      }

      //! wrapper function to allow modifying the list argument V
      template<typename T>
      auto gramSchmidt(py::list V, TensorTrain<T>& w,
                       T rankTolerance, int maxRank, bool symmetric,
                       const std::string& outputPrefix, bool verbose,
                       int nIter, bool pivoting, bool modified, bool skipDirs)
      {
        std::vector<TensorTrain<T>> Vtmp;
        for(int i = 0; i < V.size(); i++)
          Vtmp.emplace_back(std::move(V[i].cast<TensorTrain<T>&>()));
        V.attr("clear")();
        auto result = PITTS::gramSchmidt(Vtmp, w, rankTolerance, maxRank, symmetric, outputPrefix, verbose, nIter, pivoting, modified, skipDirs);
        for(int i = 0; i < Vtmp.size(); i++)
          V.append(std::move(Vtmp[i]));
        return result;
      }


      //! provide all TensorTrain<T> related classes and functions
      template<typename T>
      void init_TensorTrain_helper(py::module& m, const std::string& type_name)
      {
        const std::string className = "TensorTrain_" + type_name;

        py::class_<TensorTrain<T>>(m, className.c_str(), "Simple tensor train class")
          .def(py::init< const std::vector<int>& >(), py::arg("dimensions"), "Create TensorTrain with given dimensions")
          .def("dimensions", &TensorTrain<T>::dimensions, "tensor dimensions (immutable)")
          .def("getSubTensor", &TensorTrain_getSubTensor<T>, py::arg("d"), "Return the rank-3 sub-tensor for dimension 'd'")
          .def("setSubTensor", &TensorTrain_setSubTensor<T>, py::arg("d"), py::arg("array"), "Set the rank-3 sub-tensor for dimension 'd'")
          .def("setTTranks", py::overload_cast<int>(&TensorTrain<T>::setTTranks), py::arg("tt_rank"), "set sub-tensor dimensions (TT-ranks), destroying all existing data")
          .def("setTTranks", py::overload_cast<const std::vector<int>& >(&TensorTrain<T>::setTTranks), py::arg("tt_ranks"), "set sub-tensor dimensions (TT-ranks), destroying all existing data")
          .def("getTTranks", &TensorTrain<T>::getTTranks, "get current sub-tensor dimensions (TT-ranks)")
          .def("setZero", &TensorTrain<T>::setZero, "make this a tensor of zeros")
          .def("setOnes", &TensorTrain<T>::setOnes, "make this a tensor of ones")
          .def("setUnit", &TensorTrain<T>::setUnit, py::arg("index"), "make this a canonical unit tensor in the given direction")
          .def("__str__", &TensorTrain_toString<T>, "Print the attributes of the given TensorTrain object");

        m.def("copy",
            py::overload_cast< const TensorTrain<T>&, TensorTrain<T>& >(&copy<T>),
            py::arg("source"), py::arg("destination"),
            "explicitly copy a TensorTrain object");

        m.def("axpby",
            py::overload_cast< T, const TensorTrain<T>&, T, TensorTrain<T>&, T, int>(&axpby<T>),
            py::arg("alpha"), py::arg("TTx"), py::arg("beta"), py::arg("TTy"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), py::arg("maxRank")=std::numeric_limits<int>::max(),
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
            py::overload_cast< TensorTrain<T>&, T, int >(&normalize<T>),
            py::arg("TT"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), py::arg("maxRank")=std::numeric_limits<int>::max(),
            "TT-rounting: truncate tensor train by two normalization sweeps (first right to left, then left to right)");

        m.def("leftNormalize",
            py::overload_cast< TensorTrain<T>&, T, int >(&leftNormalize<T>),
            py::arg("TT"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), py::arg("maxRank")=std::numeric_limits<int>::max(),
            "Make all sub-tensors orthogonal sweeping left to right");

        m.def("rightNormalize",
            py::overload_cast< TensorTrain<T>&, T, int >(&rightNormalize<T>),
            py::arg("TT"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), py::arg("maxRank")=std::numeric_limits<int>::max(),
            "Make all sub-tensors orthogonal sweeping right to left");

        m.def("randomize",
            py::overload_cast< TensorTrain<T>& >(&randomize<T>),
            py::arg("TT"),
            "fill a tensor train format with random values (keeping current TT-ranks)");

        m.def("fromDense_classical",
            &TensorTrain_fromDense_classical<T>,
            py::arg("array"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), py::arg("maxRank")=-1,
            "calculate tensor-train decomposition of a tensor stored in fully dense format");

        m.def("fromDense",
            &fromDense<T>,
            py::arg("X"), py::arg("work"), py::arg("dimensions"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), py::arg("maxRank")=-1, py::arg("mpiGlobal")=false, py::arg("r0")=1, py::arg("rd")=1,
            "calculate tensor-train decomposition of a tensor stored in fully dense format (using a PITTS::MultiVector as buffer);\nWARNING: X is overwritten with temporary data to reduce memory consumption!");

        m.def("fromDense_twoSided",
            &fromDense_twoSided<T>,
            py::arg("X"), py::arg("work"), py::arg("dimensions"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), py::arg("maxRank")=-1,
            "calculate tensor-train decomposition of a tensor stored in fully dense format (using a PITTS::MultiVector as buffer);\nWARNING: X is overwritten with temporary data to reduce memory consumption!");

        m.def("toDense",
            &TensorTrain_toDense<T>,
            py::arg("TT"),
            "calculate fully dense tensor from a tensor-train decomposition");
        
        m.def("gramSchmidt",
            &pybind::gramSchmidt<T>,
            py::arg("V"), py::arg("w"),
            py::arg("rankTolerance") = std::sqrt(std::numeric_limits<T>::epsilon()), py::arg("maxRank")=std::numeric_limits<int>::max(),
            py::arg("symmetric")=false,
            py::arg("outputPrefix")="", py::arg("verbose")=false,
            py::arg("nIter")=4, py::arg("pivoting")=true, py::arg("modified")=true, py::arg("skipDirs")=true,
            "Modified Gram-Schmidt orthogonalization algorithm in Tensor-Train format");
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
