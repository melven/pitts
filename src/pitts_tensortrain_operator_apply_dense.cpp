// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_operator_apply_dense_impl.hpp"

using namespace PITTS;

template void PITTS::apply<double>(const TensorTrainOperator<double>& TTOp, const MultiVector<double>& TTx, MultiVector<double>& TTy);
template void PITTS::apply<float>(const TensorTrainOperator<float>& TTOp, const MultiVector<float>& TTx, MultiVector<float>& TTy);

template class PITTS::TTOpApplyDenseHelper<double>;
template class PITTS::TTOpApplyDenseHelper<float>;

template void PITTS::apply<double>(const TTOpApplyDenseHelper<double>& TTOp, MultiVector<double>& TTx, MultiVector<double>& TTy);
template void PITTS::apply<float>(const TTOpApplyDenseHelper<float>& TTOp, MultiVector<float>& TTx, MultiVector<float>& TTy);
