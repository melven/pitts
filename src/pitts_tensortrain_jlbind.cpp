// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_jlbind.cpp
* @brief Julia binding for PITTS::TensorTrain
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-12-18
*
**/

// includes
#include <jlcxx/jlcxx.hpp>
#include <string>
#include <stdexcept>
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
#include "pitts_tensortrain_jlbind.hpp"
#include "pitts_scope_info.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for Julia bindings
  namespace jlbind
  {
    //! internal namespace for helper functions
    namespace
    {
      /*
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
      jlcxx::array_t<T> TensorTrain_getSubTensor(const TensorTrain<T>& TT, int d)
      {
        const auto& subT = TT.subTensor(d);
        jlcxx::array_t<T> array({subT.r1(), subT.n(), subT.r2()});

        for(int i2 = 0; i2 < subT.r2(); i2++)
          for(int j = 0; j < subT.n(); j++)
            for(int i1 = 0; i1 < subT.r1(); i1++)
              *array.mutable_data(i1,j,i2) = subT(i1,j,i2);
        return array;
      }

      //! helper function to set a sub-tensor in a TensorTrain
      template<typename T>
      void TensorTrain_setSubTensor(TensorTrain<T>& TT, int d, jlcxx::array_t<T> array)
      {
        PITTS::Tensor3<T> subT(TT.subTensor(d).r1(), TT.subTensor(d).n(), TT.subTensor(d).r2());
        if( array.ndim() != 3 )
          throw std::invalid_argument("array must have 3 dimensions");
        if( array.shape(0) != subT.r1() || array.shape(1) != subT.n() || array.shape(2) != subT.r2() )
          throw std::invalid_argument("array has incompatible shape, expected " + to_string<long long>({subT.r1(), subT.n(), subT.r2()}) + ", got " + to_string<long long>({array.shape(), array.shape()+array.ndim()}));

        for(long long i2 = 0; i2 < subT.r2(); i2++)
          for(long long j = 0; j < subT.n(); j++)
            for(long long i1 = 0; i1 < subT.r1(); i1++)
              subT(i1,j,i2) = *array.data(i1,j,i2);

        TT.setSubTensor(d, std::move(subT));
      }

      //! helper function for fromDense -> TensorTrain
      template<typename T>
      TensorTrain<T> TensorTrain_fromDense_classical(jlcxx::array_t<T, jlcxx::array::f_style> array, T rankTolerance, int maxRank)
      {
        const auto first = static_cast<const T*>(array.data());
        const auto last = first + array.size();
        auto shape = std::vector<int>{array.shape(), array.shape()+array.ndim()};
        return fromDense_classical(first, last, shape, rankTolerance, maxRank);
      }

      //! helper function for TensorTrain -> toDense
      template<typename T>
      jlcxx::array_t<T, jlcxx::array::f_style> TensorTrain_toDense(const TensorTrain<T>& TT)
      {
        jlcxx::array_t<T, jlcxx::array::f_style> array(TT.dimensions());
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
      auto gramSchmidt(jlcxx::list V, TensorTrain<T>& w,
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
      void define_TensorTrain_helper(jlcxx::Module& m, const std::string& type_name)
      {
        const std::string className = "TensorTrain_" + type_name;

        jlcxx::class_<TensorTrain<T>>(m, className.c_str(), "Simple tensor train class")
          .def(jlcxx::init< const std::vector<int>& >(), jlcxx::arg("dimensions"), "Create TensorTrain with given dimensions")
          .def("dimensions", &TensorTrain<T>::dimensions, "tensor dimensions (immutable)")
          .def("getSubTensor", &TensorTrain_getSubTensor<T>, jlcxx::arg("d"), "Return the rank-3 sub-tensor for dimension 'd'")
          .def("setSubTensor", &TensorTrain_setSubTensor<T>, jlcxx::arg("d"), jlcxx::arg("array"), "Set the rank-3 sub-tensor for dimension 'd'")
          .def("setTTranks", jlcxx::overload_cast<int>(&TensorTrain<T>::setTTranks), jlcxx::arg("tt_rank"), "set sub-tensor dimensions (TT-ranks), destroying all existing data")
          .def("setTTranks", jlcxx::overload_cast<const std::vector<int>& >(&TensorTrain<T>::setTTranks), jlcxx::arg("tt_ranks"), "set sub-tensor dimensions (TT-ranks), destroying all existing data")
          .def("getTTranks", &TensorTrain<T>::getTTranks, "get current sub-tensor dimensions (TT-ranks)")
          .def("setZero", &TensorTrain<T>::setZero, "make this a tensor of zeros")
          .def("setOnes", &TensorTrain<T>::setOnes, "make this a tensor of ones")
          .def("setUnit", &TensorTrain<T>::setUnit, jlcxx::arg("index"), "make this a canonical unit tensor in the given direction")
          .def("__str__", &TensorTrain_toString<T>, "Print the attributes of the given TensorTrain object");

        m.method("cojlcxx",
            jlcxx::overload_cast< const TensorTrain<T>&, TensorTrain<T>& >(&copy<T>),
            jlcxx::arg("source"), jlcxx::arg("destination"),
            "explicitly copy a TensorTrain object");

        m.method("axpby",
            jlcxx::overload_cast< T, const TensorTrain<T>&, T, TensorTrain<T>&, T, int>(&axpby<T>),
            jlcxx::arg("alpha"), jlcxx::arg("TTx"), jlcxx::arg("beta"), jlcxx::arg("TTy"), jlcxx::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), jlcxx::arg("maxRank")=std::numeric_limits<int>::max(),
            "Scale and add one tensor train to another\n\nCalculate gamma*TTy <- alpha*TTx + beta*TTy\n\nBoth tensors must be leftNormalized, gamma is returned");

        m.method("dot",
            jlcxx::overload_cast< const TensorTrain<T>&, const TensorTrain<T>& >(&dot<T>),
            jlcxx::arg("TTx"), jlcxx::arg("TTy"),
            "calculate the inner product of two vectors in tensor train format");

        m.method("norm2",
            jlcxx::overload_cast< const TensorTrain<T>& >(&norm2<T>),
            jlcxx::arg("TT"),
            "calculate the 2-norm for a vector in tensor train format");

        m.method("normalize",
            jlcxx::overload_cast< TensorTrain<T>&, T, int >(&normalize<T>),
            jlcxx::arg("TT"), jlcxx::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), jlcxx::arg("maxRank")=std::numeric_limits<int>::max(),
            "TT-rounting: truncate tensor train by two normalization sweeps (first right to left, then left to right)");

        m.method("leftNormalize",
            jlcxx::overload_cast< TensorTrain<T>&, T, int >(&leftNormalize<T>),
            jlcxx::arg("TT"), jlcxx::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), jlcxx::arg("maxRank")=std::numeric_limits<int>::max(),
            "Make all sub-tensors orthogonal sweeping left to right");

        m.method("rightNormalize",
            jlcxx::overload_cast< TensorTrain<T>&, T, int >(&rightNormalize<T>),
            jlcxx::arg("TT"), jlcxx::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), jlcxx::arg("maxRank")=std::numeric_limits<int>::max(),
            "Make all sub-tensors orthogonal sweeping right to left");

        m.method("randomize",
            jlcxx::overload_cast< TensorTrain<T>& >(&randomize<T>),
            jlcxx::arg("TT"),
            "fill a tensor train format with random values (keeping current TT-ranks)");

        m.method("fromDense_classical",
            &TensorTrain_fromDense_classical<T>,
            jlcxx::arg("array"), jlcxx::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), jlcxx::arg("maxRank")=-1,
            "calculate tensor-train decomposition of a tensor stored in fully dense format");

        m.method("fromDense",
            &fromDense<T>,
            jlcxx::arg("X"), jlcxx::arg("work"), jlcxx::arg("dimensions"), jlcxx::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), jlcxx::arg("maxRank")=-1, jlcxx::arg("mpiGlobal")=false, jlcxx::arg("r0")=1, jlcxx::arg("rd")=1,
            "calculate tensor-train decomposition of a tensor stored in fully dense format (using a PITTS::MultiVector as buffer);\nWARNING: X is overwritten with temporary data to reduce memory consumption!");

        m.method("fromDense_twoSided",
            &fromDense_twoSided<T>,
            jlcxx::arg("X"), jlcxx::arg("work"), jlcxx::arg("dimensions"), jlcxx::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), jlcxx::arg("maxRank")=-1,
            "calculate tensor-train decomposition of a tensor stored in fully dense format (using a PITTS::MultiVector as buffer);\nWARNING: X is overwritten with temporary data to reduce memory consumption!");

        m.method("toDense",
            &TensorTrain_toDense<T>,
            jlcxx::arg("TT"),
            "calculate fully dense tensor from a tensor-train decomposition");
        
        m.method("gramSchmidt",
            &jlbind::gramSchmidt<T>,
            jlcxx::arg("V"), jlcxx::arg("w"),
            jlcxx::arg("rankTolerance") = std::sqrt(std::numeric_limits<T>::epsilon()), jlcxx::arg("maxRank")=std::numeric_limits<int>::max(),
            jlcxx::arg("symmetric")=false,
            jlcxx::arg("outputPrefix")="", jlcxx::arg("verbose")=false,
            jlcxx::arg("nIter")=4, jlcxx::arg("pivoting")=true, jlcxx::arg("modified")=true, jlcxx::arg("skipDirs")=true,
            "Modified Gram-Schmidt orthogonalization algorithm in Tensor-Train format");
      }
    */
    }

    // create jlcxx-wrapper for PITTS::TensorTrain
    void define_TensorTrain(jlcxx::Module& m)
    {
      //define_TensorTrain_helper<float>(m, "float");
      //define_TensorTrain_helper<double>(m, "double");
      //define_TensorTrain_helper<std::complex<float>>(m, "float_complex");
      //define_TensorTrain_helper<std::complex<double>>(m, "double_complex");
    }
  }
}