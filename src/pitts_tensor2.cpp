// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensor2_impl.hpp"

using namespace PITTS;

template void PITTS::copy<double>(const Tensor2<double>&, Tensor2<double>&);
