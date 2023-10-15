// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_axpby_normalized_impl.hpp"

using namespace PITTS;

template void PITTS::internal::t3_axpy<double>(const double a, const Tensor3<double>& x, Tensor3<double>& y);
template void PITTS::internal::t3_axpy<float>(const float a, const Tensor3<float>& x, Tensor3<float>& y);

template double PITTS::internal::axpby_normalized<double>(double alpha, const TensorTrain<double>& TTx, double beta, TensorTrain<double>& TTy, double rankTolerance, int maxRank);
template float PITTS::internal::axpby_normalized<float>(float alpha, const TensorTrain<float>& TTx, float beta, TensorTrain<float>& TTy, float rankTolerance, int maxRank);

template bool PITTS::internal::is_normalized<double>(const TensorTrain<double>& A, TT_Orthogonality orthog, double eps);
template bool PITTS::internal::is_normalized<float>(const TensorTrain<float>& A, TT_Orthogonality orthog, double eps);
