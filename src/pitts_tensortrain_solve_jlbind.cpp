// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_solve_jlbind.cpp
* @brief Julia binding for high-level solvers for linear system in tensor-train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-12-18
*
**/

// includes
#include <jlcxx/jlcxx.hpp>
#include <string>
#include <exception>
#include "pitts_tensortrain_solve_mals.hpp"
#include "pitts_tensortrain_solve_gmres.hpp"
#include "pitts_tensortrain_solve_jlbind.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for Julia bindings
  namespace jlbind
  {
    namespace
    {
      /*
      //! provide all TensorTrain<T> related classes and functions
      template<typename T>
      void define_TensorTrain_solve_helper(py::module& m, [[maybe_unused]] const std::string& type_name)
      {
        m.method("solveMALS",
            &solveMALS<T>,
            py::arg("TTOpA"), py::arg("symmetric"), py::arg("projection"), py::arg("TTb"), py::arg("TTx"), py::arg("nSweeps"),
            py::arg("residualTolerance")=std::numeric_limits<T>::epsilon(), py::arg("maxRank")=std::numeric_limits<int>::max(),
            py::arg("nMALS")=2, py::arg("nOverlap")=1, py::arg("nAMEnEnrichment")=0, py::arg("simplifiedAMEn")=true,
            py::arg("useTTgmres")=false, py::arg("gmresMaxIter") = 25, py::arg("gmresRelTol") = 1.e-4, py::arg("estimatedConditionTTgmres") = 10,
            "Solve a linear system using the MALS (or ALS) algorithm\n\nApproximate TTx with TTOpA * TTx = TTb");
        
        m.method("solveGMRES",
            &solveGMRES<T>,
            py::arg("TTOpA"), py::arg("TTb"), py::arg("TTx"),
            py::arg("maxIter"), py::arg("absResTol"), py::arg("relResTol"), py::arg("estimatedCondition"),
            py::arg("maxRank")=std::numeric_limits<int>::max(), py::arg("adaptiveTolerance")=true, py::arg("symmetric")=false,
            py::arg("outputPrefix")="", py::arg("verbose")=false,
            "TT-GMRES: iterative solver for linear systems in tensor-train format");
      }
      */
    }

    // create jlcxx-wrapper for PITTS::TensorTrain
    void define_TensorTrain_solve(jlcxx::Module& m)
    {
      //py::enum_<MALS_projection>(m, "MALS_projection")
      //  .value("RitzGalerkin", MALS_projection::RitzGalerkin)
      //  .value("NormalEquations", MALS_projection::NormalEquations)
      //  .value("PetrovGalerkin", MALS_projection::PetrovGalerkin)
      //  .export_values();


      //define_TensorTrain_solve_helper<float>(m, "float");
      //define_TensorTrain_solve_helper<double>(m, "double");
      //define_TensorTrain_helper<std::complex<float>>(m, "float_complex");
      //define_TensorTrain_helper<std::complex<double>>(m, "double_complex");
    }
  }
}