// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_solve_gmres_impl.hpp"

using namespace PITTS;

template double PITTS::solveGMRES(const TensorTrainOperator<double> &TTOpA, const TensorTrain<double> &TTb, TensorTrain<double> &TTx, int maxIter, double absResTol, double relResTol, int maxRank, bool adaptiveTolerance, bool symmetric, const std::string_view &outputPrefix, bool verbose);
template float PITTS::solveGMRES(const TensorTrainOperator<float> &TTOpA, const TensorTrain<float> &TTb, TensorTrain<float> &TTx, int maxIter, float absResTol, float relResTol, int maxRank, bool adaptiveTolerance, bool symmetric, const std::string_view &outputPrefix, bool verbose);