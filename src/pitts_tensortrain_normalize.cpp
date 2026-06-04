// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_normalize_impl.hpp"

using namespace PITTS;

template void PITTS::internal::leftNormalize_range<double>(TensorTrain<double>& TT, int firstIdx, int lastIdx, double rankTolerance, int maxRank);
template void PITTS::internal::leftNormalize_range<float>(TensorTrain<float>& TT, int firstIdx, int lastIdx, float rankTolerance, int maxRank);

template void PITTS::internal::rightNormalize_range<double>(TensorTrain<double>& TT, int firstIdx, int lastIdx, double rankTolerance, int maxRank);
template void PITTS::internal::rightNormalize_range<float>(TensorTrain<float>& TT, int firstIdx, int lastIdx, float rankTolerance, int maxRank);

template void PITTS::internal::ensureLeftOrtho_range<double>(TensorTrain<double>& TT, int firstIdx, int lastIdx);
template void PITTS::internal::ensureLeftOrtho_range<float>(TensorTrain<float>& TT, int firstIdx, int lastIdx);

template void PITTS::internal::ensureRightOrtho_range<double>(TensorTrain<double>& TT, int firstIdx, int lastIdx);
template void PITTS::internal::ensureRightOrtho_range<float>(TensorTrain<float>& TT, int firstIdx, int lastIdx);

template double PITTS::normalize<double>(TensorTrain<double>& TT, double rankTolerance, int maxRank);
template float PITTS::normalize<float>(TensorTrain<float>& TT, float rankTolerance, int maxRank);

template double PITTS::leftNormalize<double>(TensorTrain<double>& TT, double rankTolerance, int maxRank);
template float PITTS::leftNormalize<float>(TensorTrain<float>& TT, float rankTolerance, int maxRank);

template double PITTS::rightNormalize<double>(TensorTrain<double>& TT, double rankTolerance, int maxRank);
template float PITTS::rightNormalize<float>(TensorTrain<float>& TT, float rankTolerance, int maxRank);

template void PITTS::internal::t3_scale<double>(double, PITTS::Tensor3<double>&);
template void PITTS::internal::t3_scale<float>(float, PITTS::Tensor3<float>&);

template void PITTS::internal::normalize_contract1<double>(const Tensor2<double>& A, const Tensor3<double>& B, Tensor3<double>& C);
template void PITTS::internal::normalize_contract1<float>(const Tensor2<float>& A, const Tensor3<float>& B, Tensor3<float>& C);

template void PITTS::internal::normalize_contract2<double>(const Tensor3<double>& A, const Tensor2<double>& B, Tensor3<double>& C);
template void PITTS::internal::normalize_contract2<float>(const Tensor3<float>& A, const Tensor2<float>& B, Tensor3<float>& C);

