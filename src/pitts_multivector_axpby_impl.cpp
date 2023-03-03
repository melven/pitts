// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_axpby_impl.hpp"

using namespace PITTS;

template void PITTS::axpy<double>(const Eigen::ArrayX<double>& alpha, const MultiVector<double>& X, MultiVector<double>& Y);
template Eigen::ArrayX<double> PITTS::axpy_norm2<double>(const Eigen::ArrayX<double>& alpha, const MultiVector<double>& X, MultiVector<double>& Y);
template Eigen::ArrayX<double> PITTS::axpy_dot<double>(const Eigen::ArrayX<double>& alpha, const MultiVector<double>& X, MultiVector<double>& Y, const MultiVector<double>& Z);
