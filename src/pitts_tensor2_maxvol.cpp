// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensor2_maxvol_impl.hpp"

using namespace PITTS;

template std::vector<int> PITTS::internal::maxvol<float>(const ConstTensor2View<float>& A, float tol, int maxIter, Tensor2<float>* AAinv, int nRows);
template std::vector<int> PITTS::internal::maxvol<double>(const ConstTensor2View<double>& A, double tol, int maxIter, Tensor2<double>* AAinv, int nRows);
template std::vector<int> PITTS::internal::maxvol<std::complex<float>>(const ConstTensor2View<std::complex<float>>& A, std::complex<float> tol, int maxIter, Tensor2<std::complex<float>>* AAinv, int nRows);
template std::vector<int> PITTS::internal::maxvol<std::complex<double>>(const ConstTensor2View<std::complex<double>>& A, std::complex<double> tol, int maxIter, Tensor2<std::complex<double>>* AAinv, int nRows);
