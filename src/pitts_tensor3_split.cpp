// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensor3_split_impl.hpp"

using namespace PITTS;


template std::pair<Tensor2<double>, Tensor2<double>> PITTS::internal::normalize_qb<double>(const Tensor2<double>& M, bool leftOrthog, double rankTolerance, int maxRank, bool absoluteTolerance);
template std::pair<Tensor2<float>, Tensor2<float>> PITTS::internal::normalize_qb<float>(const Tensor2<float>& M, bool leftOrthog, float rankTolerance, int maxRank, bool absoluteTolerance);
template std::pair<Tensor2<double>, Tensor2<double>> PITTS::internal::normalize_svd<double>(const Tensor2<double>& M, bool leftOrthog, double rankTolerance, int maxRank);
template std::pair<Tensor2<float>, Tensor2<float>> PITTS::internal::normalize_svd<float>(const Tensor2<float>& M, bool leftOrthog, float rankTolerance, int maxRank);
template std::pair<Tensor3<double>, Tensor3<double>> PITTS::split<double>(const Tensor3<double>& t3c, int na, int nb, bool leftOrthog, double rankTolerance, int maxRank);
template std::pair<Tensor3<float>, Tensor3<float>> PITTS::split<float>(const Tensor3<float>& t3c, int na, int nb, bool leftOrthog, float rankTolerance, int maxRank);