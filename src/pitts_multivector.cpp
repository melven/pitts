// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_impl.hpp"

using namespace PITTS;

template class PITTS::MultiVector<double>;
template class PITTS::MultiVector<float>;

template void PITTS::copy<double>(const MultiVector<double>& a, MultiVector<double>& b);
template void PITTS::copy<float>(const MultiVector<float>& a, MultiVector<float>& b);

