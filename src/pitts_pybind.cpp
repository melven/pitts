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


PYBIND11_MODULE(pitts_py, m)
{
  PITTS::pybind::init_QubitSimulator(m);
}
