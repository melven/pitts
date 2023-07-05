// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_transform_impl.hpp"

using namespace PITTS;

template void PITTS::transform<double>(const MultiVector<double>& X, const ConstTensor2View<double>& M, MultiVector<double>& Y, std::array<long long,2> reshape);
template void PITTS::transform<float>(const MultiVector<float>& X, const ConstTensor2View<float>& M, MultiVector<float>& Y, std::array<long long,2> reshape);