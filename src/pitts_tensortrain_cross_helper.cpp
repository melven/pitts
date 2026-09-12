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
template std::vector<int> PITTS::internal::tt_cross::maxvolColumns<double>(const ConstTensor2View<double>& Vt, double maxvolTolerance, Tensor2<double>* coefficients);
template std::vector<int> PITTS::internal::tt_cross::maxvolColumns<float>(const ConstTensor2View<float>& Vt, float maxvolTolerance, Tensor2<float>* coefficients);
template CoreUpdate<double> PITTS::internal::tt_cross::updateCoreLeft<double>(const Tensor3<double>& B, double rankTolerance, int maxRank, double maxvolTolerance, optional_cref<Tensor2<double>> enrichment);
template CoreUpdate<float> PITTS::internal::tt_cross::updateCoreLeft<float>(const Tensor3<float>& B, float rankTolerance, int maxRank, float maxvolTolerance, optional_cref<Tensor2<float>> enrichment);
template CoreUpdate<double> PITTS::internal::tt_cross::updateCoreRight<double>(const Tensor3<double>& B, double rankTolerance, int maxRank, double maxvolTolerance);
template CoreUpdate<float> PITTS::internal::tt_cross::updateCoreRight<float>(const Tensor3<float>& B, float rankTolerance, int maxRank, float maxvolTolerance);
template std::vector<NestedIndices> PITTS::internal::tt_cross::initialRightIndexSets<double>(const TensorTrain<double>& TTx, double maxvolTolerance);
template std::vector<NestedIndices> PITTS::internal::tt_cross::initialRightIndexSets<float>(const TensorTrain<float>& TTx, float maxvolTolerance);
