// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_solve_mals_debug_impl.hpp"


namespace PITTS::internal::solve_mals
{
template TensorTrain<float> removeBoundaryRank(const TensorTrain<float>& tt);
template TensorTrain<double> removeBoundaryRank(const TensorTrain<double>& tt);

template TensorTrain<float> removeBoundaryRankOne(const TensorTrainOperator<float>& ttOp);
template TensorTrain<double> removeBoundaryRankOne(const TensorTrainOperator<double>& ttOp);
      
template Tensor3<float> operator-(const Tensor3<float>& a, const Tensor3<float>& b);
template Tensor3<double> operator-(const Tensor3<double>& a, const Tensor3<double>& b);
}