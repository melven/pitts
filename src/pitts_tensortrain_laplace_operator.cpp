// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_laplace_operator_impl.hpp"

using namespace PITTS;

template double PITTS::laplaceOperator<double>(TensorTrain<double>& TT, double rankTolerance);
template float PITTS::laplaceOperator<float>(TensorTrain<float>& TT, float rankTolerance);
