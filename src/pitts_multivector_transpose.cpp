// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_transpose_impl.hpp"

using namespace PITTS;

template void PITTS::transpose<double>(const MultiVector<double>& X, MultiVector<double>& Y, std::array<long long,2> reshape = {0, 0}, bool reverse = false);
template void PITTS::transpose<float>(const MultiVector<float>& X, MultiVector<float>& Y, std::array<long long,2> reshape = {0, 0}, bool reverse = false);