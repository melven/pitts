// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_operator_impl.hpp"

using namespace PITTS;

template double PITTS::normalize<double>(TensorTrainOperator<double>& TTOp, double rankTolerance, int maxRank);
template float PITTS::normalize<float>(TensorTrainOperator<float>& TTOp, float rankTolerance, int maxRank);

template void PITTS::randomize<double>(TensorTrainOperator<double>& TTOp);
template void PITTS::randomize<float>(TensorTrainOperator<float>& TTOp);

template void PITTS::axpby<double>(double alpha, const TensorTrainOperator<double>& TTOpx, double beta, TensorTrainOperator<double>& TTOpy, double rankTolerance);
template void PITTS::axpby<float>(float alpha, const TensorTrainOperator<float>& TTOpx, float beta, TensorTrainOperator<float>& TTOpy, float rankTolerance);

template void PITTS::copy<double>(const TensorTrainOperator<double>& a, TensorTrainOperator<double>& b);
template void PITTS::copy<float>(const TensorTrainOperator<float>& a, TensorTrainOperator<float>& b);