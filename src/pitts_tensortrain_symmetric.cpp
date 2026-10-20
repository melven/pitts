// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_symmetric_impl.hpp"

using namespace PITTS;
using namespace PITTS::internal::symmetric_tt;

// only real types (symmetric eigenvalue problem, see pitts_tensortrain_symmetric.hpp)
template Tensor2<double> PITTS::internal::symmetric_tt::leftOrthonormalizeHalf<double>(TensorTrain<double>& H);
template Tensor2<float> PITTS::internal::symmetric_tt::leftOrthonormalizeHalf<float>(TensorTrain<float>& H);
template Tensor2<double> PITTS::internal::symmetric_tt::halfInnerProducts<double>(const TensorTrain<double>& A, const TensorTrain<double>& B);
template Tensor2<float> PITTS::internal::symmetric_tt::halfInnerProducts<float>(const TensorTrain<float>& A, const TensorTrain<float>& B);
template SymmetricEigen<double> PITTS::internal::symmetric_tt::symmetricEigen<double>(const Tensor2<double>& C, double rankTolerance, int maxRank, bool absoluteTolerance);
template SymmetricEigen<float> PITTS::internal::symmetric_tt::symmetricEigen<float>(const Tensor2<float>& C, float rankTolerance, int maxRank, bool absoluteTolerance);
template void PITTS::internal::symmetric_tt::symmetricRound<double>(TensorTrain<double>& G, Tensor2<double>& S, double rankTolerance, int maxRank);
template void PITTS::internal::symmetric_tt::symmetricRound<float>(TensorTrain<float>& G, Tensor2<float>& S, float rankTolerance, int maxRank);
template double PITTS::internal::symmetric_tt::positiveHalf<double>(const TensorTrain<double>& G, const Tensor2<double>& S, TensorTrain<double>& H);
template float PITTS::internal::symmetric_tt::positiveHalf<float>(const TensorTrain<float>& G, const Tensor2<float>& S, TensorTrain<float>& H);
