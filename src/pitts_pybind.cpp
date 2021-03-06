/*! @file pitts_pybind.cpp
* @brief python binding for PITTS
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-26
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// includes
#include <pybind11/pybind11.h>
#include "pitts_qubit_simulator_pybind.hpp"
#include "pitts_tensortrain_pybind.hpp"
#include "pitts_tensortrain_operator_pybind.hpp"
#include "pitts_multivector_pybind.hpp"
#include "pitts_common_pybind.hpp"
#ifdef PITTS_HAVE_ITENSOR
#include "pitts_tensortrain_operator_itensor_autompo_pybind.hpp"
#endif


PYBIND11_MODULE(pitts_py, m)
{
  m.doc() = "Parallel Iterative Tensor Train Solvers (PITTS) library";

  PITTS::pybind::init_QubitSimulator(m);
  PITTS::pybind::init_TensorTrain(m);
  PITTS::pybind::init_TensorTrainOperator(m);
#ifdef PITTS_HAVE_ITENSOR
  PITTS::pybind::init_TensorTrainOperator_itensor_autompo(m);
#endif
  PITTS::pybind::init_MultiVector(m);
  PITTS::pybind::init_common(m);
}
