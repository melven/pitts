// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_solve_mals_impl.hpp"

using namespace PITTS;

template double PITTS::solveMALS(const TensorTrainOperator<double>& TTOpA, bool symmetric, const MALS_projection projection, const TensorTrain<double>& TTb, TensorTrain<double>& TTx, int nSweeps, double residualTolerance, int maxRank, int nMALS, int nOverlap, bool useTTgmres, int gmresMaxIter, double gmresRelTol);
template float PITTS::solveMALS(const TensorTrainOperator<float>& TTOpA, bool symmetric, const MALS_projection projection, const TensorTrain<float>& TTb, TensorTrain<float>& TTx, int nSweeps, float residualTolerance, int maxRank, int nMALS, int nOverlap, bool useTTgmres, int gmresMaxIter, float gmresRelTol);