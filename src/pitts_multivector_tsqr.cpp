// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_tsqr_impl.hpp"

using namespace PITTS;

template void PITTS::block_TSQR<double>(const MultiVector<double>& M, Tensor2<double>& R, int reductionFactor = 0, bool mpiGlobal = true, int colBlockingSize = 0);
template void PITTS::block_TSQR<float>(const MultiVector<float>& M, Tensor2<float>& R, int reductionFactor = 0, bool mpiGlobal = true, int colBlockingSize = 0);
template void PITTS::block_TSQR<std::complex<double>>(const MultiVector<std::complex<double>>& M, Tensor2<std::complex<double>>& R, int reductionFactor = 0, bool mpiGlobal = true, int colBlockingSize = 0);
template void PITTS::block_TSQR<std::complex<float>>(const MultiVector<std::complex<float>>& M, Tensor2<std::complex<float>>& R, int reductionFactor = 0, bool mpiGlobal = true, int colBlockingSize = 0);
