// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_tsqr_impl.hpp"

using namespace PITTS;

template void PITTS::block_TSQR<double>(const MultiVector<double>& M, Tensor2<double>& R, int reductionFactor = 0, bool mpiGlobal = true, int colBlockingSize = 0);
template void PITTS::block_TSQR<float>(const MultiVector<float>& M, Tensor2<float>& R, int reductionFactor = 0, bool mpiGlobal = true, int colBlockingSize = 0);
