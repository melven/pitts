/*! @file pitts_tensortrain_solve_pybind.cpp
* @brief python binding for high-level solvers for linear system in tensor-train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-07-06
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// includes
#include <string>
#include <stdexcept>
#include <variant>
#include "pitts_tensortrain_solve_mals.hpp"
#include "pitts_tensortrain_solve_gmres.hpp"
#include "pitts_tensortrain_solve_pybind.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"

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
    namespace
    {
      //! provide all TensorTrain<T> related classes and functions
      template<typename T>
      void init_TensorTrain_solve_helper(py::module& m, [[maybe_unused]] const std::string& type_name)
      {
        m.def("solveMALS",
            py::overload_cast< const TensorTrainOperator<T>&, bool, MALS_projection, const TensorTrain<T>&, TensorTrain<T>&, int, T, int, int, int, bool, int, T>(&solveMALS<T>),
            py::arg("TTOpA"), py::arg("symmetric"), py::arg("projection"), py::arg("TTb"), py::arg("TTx"), py::arg("nSweeps"),
            py::arg("residualTolerance")=std::numeric_limits<T>::epsilon(), py::arg("maxRank")=std::numeric_limits<int>::max(),
            py::arg("nMALS")=2, py::arg("nOverlap")=1,
            py::arg("useTTgmres")=false, py::arg("gmresMaxIter") = 25, py::arg("gmresRelTol") = 1.e-4,
            "Solve a linear system using the MALS (or ALS) algorithm\n\nApproximate TTx with TTOpA * TTx = TTb");
        
        m.def("solveGMRES",
            &solveGMRES<T>,
            py::arg("TTOpA"), py::arg("TTb"), py::arg("TTx"),
            py::arg("maxIter"), py::arg("absResTol"), py::arg("relResTol"),
            py::arg("maxRank")=std::numeric_limits<int>::max(), py::arg("adaptiveTolerance")=true, py::arg("symmetric")=false,
            py::arg("outputPrefix")="", py::arg("verbose")=false,
            "TT-GMRES: iterative solver for linear systems in tensor-train format");
      }
    }

    // create pybind11-wrapper for PITTS::TensorTrain
    void init_TensorTrain_solve(py::module& m)
    {
      py::enum_<MALS_projection>(m, "MALS_projection")
        .value("RitzGalerkin", MALS_projection::RitzGalerkin)
        .value("NormalEquations", MALS_projection::NormalEquations)
        .value("PetrovGalerkin", MALS_projection::PetrovGalerkin)
        .export_values();


      init_TensorTrain_solve_helper<float>(m, "float");
      init_TensorTrain_solve_helper<double>(m, "double");
      //init_TensorTrain_helper<std::complex<float>>(m, "float_complex");
      //init_TensorTrain_helper<std::complex<double>>(m, "double_complex");
    }
  }
}
