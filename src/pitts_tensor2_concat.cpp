// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensor2_concat_impl.hpp"

using namespace PITTS;

template void PITTS::concatLeftRight(const std::optional<ConstTensor2View<float>>& A, const std::optional<ConstTensor2View<float>>& B, Tensor2View<float> C);
template void PITTS::concatLeftRight(const std::optional<ConstTensor2View<double>>& A, const std::optional<ConstTensor2View<double>>& B, Tensor2View<double> C);

template void PITTS::concatTopBottom(const std::optional<ConstTensor2View<float>>& A, const std::optional<ConstTensor2View<float>>& B, Tensor2View<float> C);
template void PITTS::concatTopBottom(const std::optional<ConstTensor2View<double>>& A, const std::optional<ConstTensor2View<double>>& B, Tensor2View<double> C);
