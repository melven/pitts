// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_random_impl.hpp"

using namespace PITTS;

template void PITTS::randomize<double>(MultiVector<double>& X);
template void PITTS::randomize<float>(MultiVector<float>& X);
template void PITTS::randomize<std::complex<double>>(MultiVector<std::complex<double>>& X);
template void PITTS::randomize<std::complex<float>>(MultiVector<std::complex<float>>& X);