// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_dot_impl.hpp"

using namespace PITTS;

template double PITTS::dot<double>(const TensorTrain<double>& TTx, const TensorTrain<double>& TTy);
template float PITTS::dot<float>(const TensorTrain<float>& TTx, const TensorTrain<float>& TTy);

template void PITTS::internal::reverse_dot_contract1<double>(const Tensor2<double>& A, const Tensor3<double>& B, Tensor3<double>& C);
template void PITTS::internal::reverse_dot_contract1<float>(const Tensor2<float>& A, const Tensor3<float>& B, Tensor3<float>& C);

template void PITTS::internal::reverse_dot_contract2<double>(const Tensor3<double>& A, const Tensor3<double>& B, Tensor2<double>& C);
template void PITTS::internal::reverse_dot_contract2<float>(const Tensor3<float>& A, const Tensor3<float>& B, Tensor2<float>& C);

template double PITTS::internal::t3_dot<double>(const Tensor3<double>& A, const Tensor3<double>& B);
template float PITTS::internal::t3_dot<float>(const Tensor3<float>& A, const Tensor3<float>& B);

