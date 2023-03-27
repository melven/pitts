// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_multivector_triangular_solve_impl.hpp"

using namespace PITTS;

template void PITTS::triangularSolve<double>(MultiVector<double>& X, const Tensor2<double>& R, const std::vector<int>&);
template void PITTS::triangularSolve<float>(MultiVector<float>& X, const Tensor2<float>& R, const std::vector<int>&);