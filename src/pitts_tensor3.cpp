// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensor3_impl.hpp"

using namespace PITTS;

template void PITTS::copy<double>(const Tensor3<double>&, Tensor3<double>&);
template void PITTS::copy<float>(const Tensor3<float>&, Tensor3<float>&);
