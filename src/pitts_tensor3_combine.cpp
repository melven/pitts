// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include <complex>
#include "pitts_tensor3_combine_impl.hpp"

using namespace PITTS;

template Tensor3<double> PITTS::combine<double>(const Tensor3<double>& t3a, const Tensor3<double>& t3b, bool swap);
template Tensor3<std::complex<double>> PITTS::combine<std::complex<double>>(const Tensor3<std::complex<double>>& t3a, const Tensor3<std::complex<double>>& t3b, bool swap);