// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_transpose_impl.hpp"

using namespace PITTS;

template void PITTS::transpose<double>(const MultiVector<double>& X, MultiVector<double>& Y, std::array<long long,2> reshape = {0, 0}, bool reverse = false);