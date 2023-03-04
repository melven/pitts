// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_normalize_impl.hpp"

using namespace PITTS;

template void PITTS::internal::leftNormalize_range<double>(TensorTrain<double>& TT, int firstIdx, int lastIdx, double rankTolerance, int maxRank);
template void PITTS::internal::leftNormalize_range<float>(TensorTrain<float>& TT, int firstIdx, int lastIdx, float rankTolerance, int maxRank);

template void PITTS::internal::rightNormalize_range<double>(TensorTrain<double>& TT, int firstIdx, int lastIdx, double rankTolerance, int maxRank);
template void PITTS::internal::rightNormalize_range<float>(TensorTrain<float>& TT, int firstIdx, int lastIdx, float rankTolerance, int maxRank);

template void PITTS::internal::ensureLeftOrtho_range<double>(TensorTrain<double>& TT, int firstIdx, int lastIdx);
template void PITTS::internal::ensureLeftOrtho_range<float>(TensorTrain<float>& TT, int firstIdx, int lastIdx);

template void PITTS::internal::ensureRightOrtho_range<double>(TensorTrain<double>& TT, int firstIdx, int lastIdx);
template void PITTS::internal::ensureRightOrtho_range<float>(TensorTrain<float>& TT, int firstIdx, int lastIdx);

template double PITTS::normalize<double>(TensorTrain<double>& TT, double rankTolerance, int maxRank);
template float PITTS::normalize<float>(TensorTrain<float>& TT, float rankTolerance, int maxRank);

template double PITTS::leftNormalize<double>(TensorTrain<double>& TT, double rankTolerance, int maxRank);
template float PITTS::leftNormalize<float>(TensorTrain<float>& TT, float rankTolerance, int maxRank);

template double PITTS::rightNormalize<double>(TensorTrain<double>& TT, double rankTolerance, int maxRank);
template float PITTS::rightNormalize<float>(TensorTrain<float>& TT, float rankTolerance, int maxRank);

template void PITTS::internal::t3_scale<double>(double, PITTS::Tensor3<double>&);
template void PITTS::internal::t3_scale<float>(float, PITTS::Tensor3<float>&);
