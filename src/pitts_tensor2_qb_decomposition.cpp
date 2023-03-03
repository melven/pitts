// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensor2_qb_decomposition_impl.hpp"

using namespace PITTS;

template int PITTS::qb_decomposition<double>(const Tensor2<double>& M, Tensor2<double>& B, Tensor2<double>& Binv, double rankTolerance,  int maxRank = std::numeric_limits<int>::max(), bool absoluteTolerance = false);
