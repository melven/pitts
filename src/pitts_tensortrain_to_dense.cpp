// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_to_dense_impl.hpp"

using namespace PITTS;

template void PITTS::toDense<double>(const TensorTrain<double>& TT, MultiVector<double>& X);
template void PITTS::toDense<float>(const TensorTrain<float>& TT, MultiVector<float>& X);

template<typename T>
using Iter = T*;

template void PITTS::toDense<double, Iter<double>>(const TensorTrain<double>& TT, Iter<double>, Iter<double>);
template void PITTS::toDense<float, Iter<float>>(const TensorTrain<float>& TT, Iter<float>, Iter<float>);

template<typename T>
using StdIter = decltype(std::vector<T>().begin());

template void PITTS::toDense<double, StdIter<double>>(const TensorTrain<double>& TT, StdIter<double>, StdIter<double>);
template void PITTS::toDense<float, StdIter<float>>(const TensorTrain<float>& TT, StdIter<float>, StdIter<float>);
