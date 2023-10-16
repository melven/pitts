// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_axpby_impl.hpp"

using namespace PITTS;

template void PITTS::axpy<double>(const Eigen::ArrayX<double>& alpha, const MultiVector<double>& X, MultiVector<double>& Y);
template void PITTS::axpy<float>(const Eigen::ArrayX<float>& alpha, const MultiVector<float>& X, MultiVector<float>& Y);
template Eigen::ArrayX<double> PITTS::axpy_norm2<double>(const Eigen::ArrayX<double>& alpha, const MultiVector<double>& X, MultiVector<double>& Y);
template Eigen::ArrayX<float> PITTS::axpy_norm2<float>(const Eigen::ArrayX<float>& alpha, const MultiVector<float>& X, MultiVector<float>& Y);
template Eigen::ArrayX<double> PITTS::axpy_dot<double>(const Eigen::ArrayX<double>& alpha, const MultiVector<double>& X, MultiVector<double>& Y, const MultiVector<double>& Z);
template Eigen::ArrayX<float> PITTS::axpy_dot<float>(const Eigen::ArrayX<float>& alpha, const MultiVector<float>& X, MultiVector<float>& Y, const MultiVector<float>& Z);
