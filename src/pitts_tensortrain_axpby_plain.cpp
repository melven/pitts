// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_axpby_plain_impl.hpp"

using namespace PITTS;

template double PITTS::internal::axpby_plain<double>(double alpha, const TensorTrain<double>& TTx, double beta, TensorTrain<double>& TTy, double rankTolerance, int maxRank);
template float PITTS::internal::axpby_plain<float>(float alpha, const TensorTrain<float>& TTx, float beta, TensorTrain<float>& TTy, float rankTolerance, int maxRank);