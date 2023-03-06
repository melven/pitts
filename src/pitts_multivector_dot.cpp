// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_dot_impl.hpp"

using namespace PITTS;

template Eigen::ArrayX<double> PITTS::dot<double>(const MultiVector<double>& X, const MultiVector<double>& Y);
template Eigen::ArrayX<float> PITTS::dot<float>(const MultiVector<float>& X, const MultiVector<float>& Y);