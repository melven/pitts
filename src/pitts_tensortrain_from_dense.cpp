// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_from_dense_impl.hpp"

using namespace PITTS;

template TensorTrain<double> PITTS::fromDense<double>(MultiVector<double>& X, MultiVector<double>& work, const std::vector<int>& dimensions, double rankTolerance , int maxRank, bool, int, int);
template TensorTrain<float> PITTS::fromDense<float>(MultiVector<float>& X, MultiVector<float>& work, const std::vector<int>& dimensions, float rankTolerance , int maxRank, bool, int, int);