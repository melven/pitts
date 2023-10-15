// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_operator_to_dense_impl.hpp"

using namespace PITTS;

template Tensor2<double> PITTS::toDense<double>(const TensorTrainOperator<double>& TTOp);
template Tensor2<float> PITTS::toDense<float>(const TensorTrainOperator<float>& TTOp);
