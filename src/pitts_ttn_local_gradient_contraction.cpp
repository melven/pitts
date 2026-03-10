// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_ttn_local_gradient_contraction_impl.hpp"

using namespace PITTS;

template void PITTS::ttn_local_gradient_contract<double>(const Tensor3<double>& optTensor, const MultiVector<double>& envLeft, const MultiVector<double>& envRight,  const MultiVector<double>& envTop, Tensor3<double>& gradOptTensor, bool mpiParallel);
template void PITTS::ttn_local_gradient_contract<float>(const Tensor3<float>& optTensor, const MultiVector<float>& envLeft, const MultiVector<float>& envRight,  const MultiVector<float>& envTop, Tensor3<float>& gradOptTensor, bool mpiParallel);
