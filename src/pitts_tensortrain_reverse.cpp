// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_reverse_impl.hpp"

using namespace PITTS;

template void PITTS::internal::t3_transpose<double>(const Tensor3<double>& A, Tensor3<double>& B);
template void PITTS::internal::t3_transpose<float>(const Tensor3<float>& A, Tensor3<float>& B);

template void PITTS::reverse<double>(const TensorTrain<double>& TTx, TensorTrain<double>& TTy);
template void PITTS::reverse<float>(const TensorTrain<float>& TTx, TensorTrain<float>& TTy);
