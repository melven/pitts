// Copyright (c) 2023 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_jlbind.hpp
* @brief Julia binding for pitts
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-12-18
*
**/

#include <jlcxx/jlcxx.hpp>

#include "pitts_qubit_simulator_jlbind.hpp"
#include "pitts_tensortrain_jlbind.hpp"
#include "pitts_tensortrain_operator_jlbind.hpp"
#include "pitts_tensortrain_solve_jlbind.hpp"
#include "pitts_tensor2_jlbind.hpp"
#include "pitts_multivector_jlbind.hpp"
#include "pitts_common_jlbind.hpp"

//struct MyStruct { MyStruct() {} };

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
  //mod.add_type<MyStruct>("MyStruct");
  //PITTS::jlbind::define_QubitSimulator(mod);
  //PITTS::jlbind::define_TensorTrain(mod);
  //PITTS::jlbind::define_TensorTrainOperator(mod);
  //PITTS::jlbind::define_TensorTrain_solve(mod);
  PITTS::jlbind::define_Tensor2(mod);
  PITTS::jlbind::define_MultiVector(mod);
  PITTS::jlbind::define_common(mod);
}
