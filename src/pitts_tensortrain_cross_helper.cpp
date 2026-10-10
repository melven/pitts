// Copyright (c) 2026 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Claude Code (Anthropic AI model), prompted by Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_cross_helper_impl.hpp"

using namespace PITTS;
using namespace PITTS::internal::tt_cross;

// only float and double: the underlying TT contractions (normalize_contract*, normalize_svd, ...) are instantiated for these
template void PITTS::internal::tt_cross::enrichFactors<double>(Tensor3<double>& Q, Tensor2<double>& B, Tensor3<double>&& E, int maxDirections, double relativeTolerance);
template void PITTS::internal::tt_cross::enrichFactors<float>(Tensor3<float>& Q, Tensor2<float>& B, Tensor3<float>&& E, int maxDirections, float relativeTolerance);
template std::vector<int> PITTS::internal::tt_cross::maxvolColumns<double>(const ConstTensor2View<double>& Vt, double maxvolTolerance, Tensor2<double>* coefficients, int nColumns);
template std::vector<int> PITTS::internal::tt_cross::maxvolColumns<float>(const ConstTensor2View<float>& Vt, float maxvolTolerance, Tensor2<float>* coefficients, int nColumns);
template CoreUpdate<double> PITTS::internal::tt_cross::updateCoreLeft<double>(const Tensor3<double>& C, const Tensor2<double>& leftInterface, double rankTolerance, int maxRank, double maxvolTolerance, Tensor3<double> enrichment, double oversampling, int maxEnrichment, int maxEnrichedRank, int maxRows);
template CoreUpdate<float> PITTS::internal::tt_cross::updateCoreLeft<float>(const Tensor3<float>& C, const Tensor2<float>& leftInterface, float rankTolerance, int maxRank, float maxvolTolerance, Tensor3<float> enrichment, float oversampling, int maxEnrichment, int maxEnrichedRank, int maxRows);
template CoreUpdate<double> PITTS::internal::tt_cross::updateCoreRight<double>(const Tensor3<double>& C, const Tensor2<double>& rightInterface, double rankTolerance, int maxRank, double maxvolTolerance, double oversampling, int maxColumns);
template CoreUpdate<float> PITTS::internal::tt_cross::updateCoreRight<float>(const Tensor3<float>& C, const Tensor2<float>& rightInterface, float rankTolerance, int maxRank, float maxvolTolerance, float oversampling, int maxColumns);
template void PITTS::internal::tt_cross::solveLeftInterface<double>(optional_cref<Tensor3<double>> LX, const std::vector<NestedIndices>& I, int k, const Tensor3<double>& B, Tensor3<double>& C);
template void PITTS::internal::tt_cross::solveRightInterface<double>(const Tensor3<double>& B, optional_cref<Tensor3<double>> XR, const std::vector<NestedIndices>& J, int k, Tensor3<double>& C);
template void PITTS::internal::tt_cross::initialRightFrame<double>(const std::vector<std::reference_wrapper<const TensorTrain<double>>>& TTx, TensorTrain<double>& TTy, int maxRank);
template void PITTS::internal::tt_cross::solveLeftInterface<float>(optional_cref<Tensor3<float>> LX, const std::vector<NestedIndices>& I, int k, const Tensor3<float>& B, Tensor3<float>& C);
template void PITTS::internal::tt_cross::solveRightInterface<float>(const Tensor3<float>& B, optional_cref<Tensor3<float>> XR, const std::vector<NestedIndices>& J, int k, Tensor3<float>& C);
template void PITTS::internal::tt_cross::initialRightFrame<float>(const std::vector<std::reference_wrapper<const TensorTrain<float>>>& TTx, TensorTrain<float>& TTy, int maxRank);
template std::vector<int> PITTS::internal::tt_cross::dominantRows<double>(const Tensor3<double>& B, double rankTolerance, int maxRank, double maxvolTolerance);
template std::vector<int> PITTS::internal::tt_cross::dominantColumns<double>(const Tensor3<double>& B, double rankTolerance, int maxRank, double maxvolTolerance);
template std::vector<NestedIndices> PITTS::internal::tt_cross::rightIndexSetsFromFrame<double>(const TensorTrain<double>& TT, double maxvolTolerance, double oversampling, const std::vector<int>& maxIndices);
template std::vector<int> PITTS::internal::tt_cross::dominantRows<float>(const Tensor3<float>& B, float rankTolerance, int maxRank, float maxvolTolerance);
template std::vector<int> PITTS::internal::tt_cross::dominantColumns<float>(const Tensor3<float>& B, float rankTolerance, int maxRank, float maxvolTolerance);
template std::vector<NestedIndices> PITTS::internal::tt_cross::rightIndexSetsFromFrame<float>(const TensorTrain<float>& TT, float maxvolTolerance, float oversampling, const std::vector<int>& maxIndices);
