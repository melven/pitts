// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_operator_apply_impl.hpp"

using namespace PITTS;

template void PITTS::apply<double>(const TensorTrainOperator<double>& TTOp, const TensorTrain<double>& TTx, TensorTrain<double>& TTy);
template void PITTS::apply<float>(const TensorTrainOperator<float>& TTOp, const TensorTrain<float>& TTx, TensorTrain<float>& TTy);

template void PITTS::internal::apply_contract<double>([[maybe_unused]] const TensorTrainOperator<double>& TTOp, [[maybe_unused]] int iDim, const Tensor3<double>& Aop, const Tensor3<double>& x, Tensor3<double>& y);
template void PITTS::internal::apply_contract<float>([[maybe_unused]] const TensorTrainOperator<float>& TTOp, [[maybe_unused]] int iDim, const Tensor3<float>& Aop, const Tensor3<float>& x, Tensor3<float>& y);
