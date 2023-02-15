/*! @file pitts_tensortrain_operator_itensor_autompo_pybind.cpp
* @brief header for python binding of the PITTS wrapper for ITensor autompo
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-01-19
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// workaround ITensor / Eigen LAPACK definition problems
#ifdef EIGEN_USE_LAPACKE
#undef EIGEN_USE_LAPACKE
#endif

// includes
#include <string>
#include <stdexcept>
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_from_itensor.hpp"
#include "pitts_tensortrain_operator_itensor_autompo_pybind.hpp"
#include "itensor/all.h"

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
      //! helper function to wrap itensor::AutoMPO += functionality
      itensor::AutoMPO& ampo_add(itensor::AutoMPO& ampo, py::tuple args)
      {
        if( py::len(args) == 2 )
          ampo += py::cast<std::string>(args[0]), py::cast<int>(args[1]);
        else if( py::len(args) == 3 )
          ampo += py::cast<itensor::Complex>(args[0]), py::cast<std::string>(args[1]), py::cast<int>(args[2]);
        else if( py::len(args) == 4 )
          ampo += py::cast<std::string>(args[0]), py::cast<int>(args[1]), py::cast<std::string>(args[2]), py::cast<int>(args[3]);
        else if( py::len(args) == 5 )
          ampo += py::cast<itensor::Complex>(args[0]), py::cast<std::string>(args[1]), py::cast<int>(args[2]), py::cast<std::string>(args[3]), py::cast<int>(args[4]);
        else
          throw std::invalid_argument("Wrong number of arguments / tuple size, need tuple([Coef,] Op, i [, Op, j])!");

        return ampo;
      }

      //! helper function to create a PITTS::TensorTrainOperator from the itensor::AutoMPO
      PITTS::TensorTrainOperator<double> toTTOp(const itensor::AutoMPO& ampo)
      {
        return fromITensor<double>(itensor::toMPO(ampo));
      }
    }

    // create pybind11-wrapper for PITTS::TensorTrain
    void init_TensorTrainOperator_itensor_autompo(py::module& parent_module)
    {
      auto m = parent_module.def_submodule("itensor", "Helper functionality based on ITensor");

      // ITensor siteset classes
      py::class_<itensor::SiteSet>(m, "SiteSet", "Collection of site objects, defining a Hilbert space and local operators");
      py::class_<itensor::SpinHalf, itensor::SiteSet>(m, "SpinHalf", "S=1/2 spin sites")
        .def(py::init<int>(), py::arg("N"));
      py::class_<itensor::SpinOne, itensor::SiteSet>(m, "SpinOne", "S=1 spin sites")
        .def(py::init<int>(), py::arg("N"));
      py::class_<itensor::Boson, itensor::SiteSet>(m, "Boson", "Spinless boson sites with adjustable max occupancy")
        .def(py::init<int>(), py::arg("N"));
      py::class_<itensor::Fermion, itensor::SiteSet>(m, "Fermion", "Spinless fermion sites")
        .def(py::init<int>(), py::arg("N"));
      py::class_<itensor::Electron, itensor::SiteSet>(m, "Electron", "Spinful fermion sites")
        .def(py::init<int>(), py::arg("N"));
      py::class_<itensor::tJ, itensor::SiteSet>(m, "tJ", "t-J model sites")
        .def(py::init<int>(), py::arg("N"));

      // ITensor AutoMPO class
      py::class_<itensor::AutoMPO>(m, "AutoMPO", "System for making MPOs from sums of local operators")
        .def(py::init<const itensor::SiteSet&>(), py::arg("sites"))
        .def("__iadd__", ampo_add, py::is_operator());

      // conversion from AutoMPO type to TT operator
      m.def("toTTOp", &toTTOp, py::arg("ampo"));
    }

  }
}
