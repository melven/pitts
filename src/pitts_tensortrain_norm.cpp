// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_norm_impl.hpp"

using namespace PITTS;

template double PITTS::norm2<double>(const TensorTrain<double>& TTx);
template float PITTS::norm2<float>(const TensorTrain<float>& TTx);

template double PITTS::internal::t3_nrm<double>(const PITTS::Tensor3<double>&);
template float PITTS::internal::t3_nrm<float>(const PITTS::Tensor3<float>&);
