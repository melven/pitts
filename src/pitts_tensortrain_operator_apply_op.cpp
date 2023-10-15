// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_operator_apply_op_impl.hpp"

using namespace PITTS;

template void PITTS::apply<double>(const TensorTrainOperator<double>& TTOp, const TensorTrainOperator<double>& TTx, TensorTrainOperator<double>& TTy);
template void PITTS::apply<float>(const TensorTrainOperator<float>& TTOp, const TensorTrainOperator<float>& TTx, TensorTrainOperator<float>& TTy);
