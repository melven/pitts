// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_impl.hpp"

using namespace PITTS;

template class PITTS::MultiVector<double>;
template class PITTS::MultiVector<float>;
template class PITTS::MultiVector<std::complex<double>>;
template class PITTS::MultiVector<std::complex<float>>;

template void PITTS::copy<double>(const MultiVector<double>& a, MultiVector<double>& b);
template void PITTS::copy<float>(const MultiVector<float>& a, MultiVector<float>& b);
template void PITTS::copy<std::complex<double>>(const MultiVector<std::complex<double>>& a, MultiVector<std::complex<double>>& b);
template void PITTS::copy<std::complex<float>>(const MultiVector<std::complex<float>>& a, MultiVector<std::complex<float>>& b);
