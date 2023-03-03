// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_centroids_impl.hpp"

using namespace PITTS;

template void PITTS::centroids<double>(const MultiVector<double>& X, const std::vector<long long>& idx, const std::vector<double>& w, MultiVector<double>& Y);
