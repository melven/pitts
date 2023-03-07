// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_from_dense_classical_impl.hpp"

using namespace PITTS;

template<typename T>
using Iter = T const*;

template TensorTrain<double> PITTS::fromDense_classical<Iter<double>>(const Iter<double> first, const Iter<double> last, const std::vector<int>& dimensions, double rankTolerance , int maxRank);
template TensorTrain<float> PITTS::fromDense_classical<Iter<float>>(const Iter<float> first, const Iter<float> last, const std::vector<int>& dimensions, float rankTolerance , int maxRank);

template<typename T>
using StdIter = decltype(std::vector<T>().cbegin());

template TensorTrain<double> PITTS::fromDense_classical<StdIter<double>>(const StdIter<double> first, const StdIter<double> last, const std::vector<int>& dimensions, double rankTolerance , int maxRank);
template TensorTrain<float> PITTS::fromDense_classical<StdIter<float>>(const StdIter<float> first, const StdIter<float> last, const std::vector<int>& dimensions, float rankTolerance , int maxRank);
