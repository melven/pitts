// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_operator_apply_impl.hpp"

using namespace PITTS;

template void PITTS::apply<double>(const TensorTrainOperator<double>& TTOp, const TensorTrain<double>& TTx, TensorTrain<double>& TTy);
template void PITTS::apply<float>(const TensorTrainOperator<float>& TTOp, const TensorTrain<float>& TTx, TensorTrain<float>& TTy);