// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_cdist_impl.hpp"

using namespace PITTS;

template void PITTS::cdist2<double>(const MultiVector<double>& X, const MultiVector<double>& Y, Tensor2<double>& D);
template void PITTS::cdist2<float>(const MultiVector<float>& X, const MultiVector<float>& Y, Tensor2<float>& D);
