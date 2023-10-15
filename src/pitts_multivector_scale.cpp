// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_scale_impl.hpp"

using namespace PITTS;

template void PITTS::scale<double>(const Eigen::ArrayX<double>& alpha, MultiVector<double>& X);
template void PITTS::scale<float>(const Eigen::ArrayX<float>& alpha, MultiVector<float>& X);