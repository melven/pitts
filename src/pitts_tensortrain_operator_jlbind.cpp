// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_operator_jlbind.cpp
* @brief Julia binding for PITTS::TensorTrainOperator
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-12-18
*
**/

// includes
#include <jlcxx/jlcxx.hpp>
#include <string>
#include <stdexcept>
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply.hpp"
#include "pitts_tensortrain_operator_apply_transposed.hpp"
#include "pitts_tensortrain_operator_apply_op.hpp"
#include "pitts_tensortrain_operator_apply_transposed_op.hpp"
#include "pitts_tensortrain_operator_to_dense.hpp"
#include "pitts_tensortrain_operator_to_qtt.hpp"
#include "pitts_tensortrain_operator_jlbind.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
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

      //! helper function to obtain a sub-tensor from a TensorTrainOperator
      template<typename T>
      py::array_t<T> TensorTrainOperator_getSubTensor(const TensorTrainOperator<T>& TTOp, int d)
      {
        const auto& subT = TTOp.tensorTrain().subTensor(d);
        py::array_t<T> array({(int)subT.r1(), TTOp.row_dimensions().at(d), TTOp.column_dimensions().at(d), (int)subT.r2()});

        for(int i = 0; i < subT.r1(); i++)
          for(int j = 0; j < TTOp.row_dimensions()[d]; j++)
            for(int k = 0; k < TTOp.column_dimensions()[d]; k++)
              for(int l = 0; l < subT.r2(); l++)
                *array.mutable_data(i,j,k,l) = subT(i,TTOp.index(d, j, k), l);
        return array;
      }

      //! helper function to set a sub-tensor in a TensorTrainOperator
      template<typename T>
      void TensorTrainOperator_setSubTensor(TensorTrainOperator<T>& TTOp, int d, py::array_t<T> array)
      {
        Tensor3<T> subT(TTOp.tensorTrain().subTensor(d).r1(), TTOp.tensorTrain().subTensor(d).n(), TTOp.tensorTrain().subTensor(d).r2());
        if( array.ndim() != 4 )
          throw std::invalid_argument("array must have 4 dimensions");
        const std::vector<long long> required_shape = {subT.r1(), TTOp.row_dimensions().at(d), TTOp.column_dimensions().at(d), subT.r2()};
        const std::vector<long long> shape = {array.shape(), array.shape()+array.ndim()};
        if( shape != required_shape )
          throw std::invalid_argument("array has incompatible shape, expected " + to_string<long long>(required_shape) + ", got " + to_string<long long>(shape));

        for(int i = 0; i < shape[0]; i++)
          for(int j = 0; j < shape[1]; j++)
            for(int k = 0; k < shape[2]; k++)
              for(int l = 0; l < shape[3]; l++)
                subT(i,TTOp.index(d, j, k), l) = *array.data(i,j,k,l);
        TTOp.tensorTrain().setSubTensor(d, std::move(subT));
      }

      //! helper function to print the attributes of the TensorTrainOperator object nicely
      template<typename T>
      std::string TensorTrainOperator_toString(const TensorTrainOperator<T>& TT)
      {
        // helper for getting the template type nicely formatted
        constexpr auto scope = internal::ScopeInfo::current<T>();

        std::string result;
        result += "PITTS::TensorTrainOperator" + std::string(scope.type_name()) + "\n";
        result += "        with row dimensions = " + to_string(TT.row_dimensions()) + "\n";
        result += "             col dimensions = " + to_string(TT.column_dimensions()) + "\n";
        result += "                 ranks      = " + to_string(TT.getTTranks());

        return result;
      }

      //! provide all TensorTrain<T> related classes and functions
      template<typename T>
      void define_TensorTrainOperator_helper(py::module& m, const std::string& type_name)
      {
        const std::string className = "TensorTrainOperator_" + type_name;

        py::class_<TensorTrainOperator<T>>(m, className.c_str(), "Simple tensor train operator class")
          .def(py::init< const std::vector<int>& , const std::vector<int>& >(), py::arg("row_dimensions"), py::arg("col_dimensions"), "Create TensorTrainOperator with given dimensions")
          .def("row_dimensions", &TensorTrainOperator<T>::row_dimensions, "output tensor dimensions (immutable)")
          .def("col_dimensions", &TensorTrainOperator<T>::column_dimensions, "input tensor dimensions (immutable)")
          .def("getSubTensor", &TensorTrainOperator_getSubTensor<T>, py::arg("d"), "Return the rank-4 sub-tensor for dimension 'd'")
          .def("setSubTensor", &TensorTrainOperator_setSubTensor<T>, py::arg("d"), py::arg("array"), "Set the rank-4 sub-tensor for dimension 'd'")
          .def("setTTranks", py::overload_cast<int>(&TensorTrainOperator<T>::setTTranks), py::arg("tt_rank"), "set sub-tensor dimensions (TT-ranks), destroying all existing data")
          .def("setTTranks", py::overload_cast<const std::vector<int>& >(&TensorTrainOperator<T>::setTTranks), py::arg("tt_ranks"), "set sub-tensor dimensions (TT-ranks), destroying all existing data")
          .def("getTTranks", &TensorTrainOperator<T>::getTTranks, "get current sub-tensor dimensions (TT-ranks)")
          .def("setZero", &TensorTrainOperator<T>::setZero, "make this a tensor of zeros")
          .def("setOnes", &TensorTrainOperator<T>::setOnes, "make this a tensor of ones")
          .def("setEye", &TensorTrainOperator<T>::setEye, "make this an identity operator (if square)")
          .def("__str__", &TensorTrainOperator_toString<T>, "Print the attributes of the given TensorTrainOperator object");

        m.method("copy",
            py::overload_cast< const TensorTrainOperator<T>&, TensorTrainOperator<T>& >(&copy<T>),
            py::arg("source"), py::arg("destination"),
            "explicitly copy a TensorTrainOperator object");

        m.method("axpby",
            py::overload_cast< T, const TensorTrainOperator<T>&, T, TensorTrainOperator<T>&, T>(&axpby<T>),
            py::arg("alpha"), py::arg("TTOpx"), py::arg("beta"), py::arg("TTOpy"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()),
            "Scale and add one tensor train operator to another\n\nCalculate TTOpy <- alpha*TTOpx + beta*TTOpy\n\nBoth tensors must be leftNormalized");

        m.method("randomize",
            py::overload_cast< TensorTrainOperator<T>& >(&randomize<T>),
            py::arg("TT"),
            "fill a tensor train operator format with random values (keeping current TT-ranks)");

        m.method("normalize",
            py::overload_cast< TensorTrainOperator<T>&, T, int >(&normalize<T>),
            py::arg("TT"), py::arg("rankTolerance")=std::sqrt(std::numeric_limits<T>::epsilon()), py::arg("maxRank")=std::numeric_limits<int>::max(),
            "TT-rounting: truncate tensor train by two normalization sweeps (first right to left, then left to right)");

        m.method("apply",
            py::overload_cast< const TensorTrainOperator<T>&, const TensorTrain<T>&, TensorTrain<T>& >(&apply<T>),
            py::arg("TTOp"), py::arg("TTx"), py::arg("TTy"),
            "Apply a tensor train operator\n\nCalculate TTy <- TTOp * TTx");

        m.method("apply",
            py::overload_cast< const TensorTrainOperator<T>&, const TensorTrainOperator<T>&, TensorTrainOperator<T>& >(&apply<T>),
            py::arg("TTOpA"), py::arg("TTOpB"), py::arg("TTOpC"),
            "Apply a TT operator to another TT operator\n\nCalculate TTOpC <- TTOpA * TTOpB");

        m.method("applyT",
            py::overload_cast< const TensorTrainOperator<T>&, const TensorTrain<T>&, TensorTrain<T>& >(&applyT<T>),
            py::arg("TTOp"), py::arg("TTx"), py::arg("TTy"),
            "Apply a transposed tensor train operator\n\nCalculate TTy <- TTOp^T * TTx");

        m.method("applyT",
            py::overload_cast< const TensorTrainOperator<T>&, const TensorTrainOperator<T>&, TensorTrainOperator<T>& >(&applyT<T>),
            py::arg("TTOpA"), py::arg("TTOpB"), py::arg("TTOpC"),
            "Apply a transposed TT operator to another TT operator\n\nCalculate TTOpC <- TTOpA^T * TTOpB");
        
        m.method("toDense",
            [](const TensorTrainOperator<T>& TTOp) -> Eigen::MatrixX<T>
            {
              Tensor2<T> op = toDense(TTOp);
              // requires an additional copy for now...
              return ConstEigenMap(op);
            },
            py::arg("TTOp"),
            "calculate fully dense matrix from a tensor-train operator decomposition");
        m.method("toQtt",
            py::overload_cast<const TensorTrainOperator<T>&>(&toQtt<T>),
            py::arg("TTOp"),
            "calculate quantized tensor-train (QTT) format from a given tensor-train operator");
      }
    */
    }

    // create jlcxx-wrapper for PITTS::TensorTrain
    void define_TensorTrainOperator(jlcxx::Module& mod)
    {
      //define_TensorTrainOperator_helper<float>(m, "float");
      //define_TensorTrainOperator_helper<double>(m, "double");
      //define_TensorTrain_helper<std::complex<float>>(m, "float_complex");
      //define_TensorTrain_helper<std::complex<double>>(m, "double_complex");
    }
  }
}
