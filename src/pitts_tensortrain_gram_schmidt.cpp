// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_gram_schmidt_impl.hpp"

using namespace PITTS;


template Eigen::ArrayX<double> PITTS::gramSchmidt<double>(std::vector<TensorTrain<double>>& V, TensorTrain<double>& w, double rankTolerance, int maxRank, bool symmetric, const std::string_view& outputPrefix, bool verbose, int nIter, bool pivoting, bool modified, bool skipDirs);
template Eigen::ArrayX<float> PITTS::gramSchmidt<float>(std::vector<TensorTrain<float>>& V, TensorTrain<float>& w, float rankTolerance, int maxRank, bool symmetric, const std::string_view& outputPrefix, bool verbose, int nIter, bool pivoting, bool modified, bool skipDirs);