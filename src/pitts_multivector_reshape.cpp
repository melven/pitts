// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_reshape_impl.hpp"

using namespace PITTS;

template void PITTS::reshape<double>(const MultiVector<double>& X, long long rows, long long cols, MultiVector<double>& Y);
template void PITTS::reshape<float>(const MultiVector<float>& X, long long rows, long long cols, MultiVector<float>& Y);