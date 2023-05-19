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

template bool check_Ax(const TensorTrainOperator<float>& TTOpA, const TensorTrain<float>& TTx, SweepIndex swpIdx, const std::vector<Tensor3<float>>& Ax);
template bool check_Ax(const TensorTrainOperator<double>& TTOpA, const TensorTrain<double>& TTx, SweepIndex swpIdx, const std::vector<Tensor3<double>>& Ax);

template bool check_Ax_ortho(const TensorTrainOperator<float>& TTOpA, const TensorTrain<float>& TTx, const std::vector<std::pair<Tensor3<float>,Tensor2<float>>>& Ax_ortho);
template bool check_Ax_ortho(const TensorTrainOperator<double>& TTOpA, const TensorTrain<double>& TTx, const std::vector<std::pair<Tensor3<double>,Tensor2<double>>>& Ax_ortho);

template bool check_ProjectionOperator(const TensorTrainOperator<float>& TTOpA, const TensorTrain<float>& TTx, SweepIndex swpIdx, const TensorTrainOperator<float>& TTv, const TensorTrainOperator<float>& TTAv);
template bool check_ProjectionOperator(const TensorTrainOperator<double>& TTOpA, const TensorTrain<double>& TTx, SweepIndex swpIdx, const TensorTrainOperator<double>& TTv, const TensorTrainOperator<double>& TTAv);

template bool check_Orthogonality(SweepIndex swpIdx, const TensorTrain<float>& TTw);
template bool check_Orthogonality(SweepIndex swpIdx, const TensorTrain<double>& TTw);

template bool check_systemDimensions(const TensorTrainOperator<float>& localTTOp, const TensorTrain<float>& tt_x, const TensorTrain<float>& tt_b);
template bool check_systemDimensions(const TensorTrainOperator<double>& localTTOp, const TensorTrain<double>& tt_x, const TensorTrain<double>& tt_b);

template bool check_localProblem(const TensorTrainOperator<float>& TTOpA, const TensorTrain<float>& TTx, const TensorTrain<float>& TTb, const TensorTrain<float>& TTw, 
                                 bool ritzGalerkinProjection, SweepIndex swpIdx,
                                 const TensorTrainOperator<float>& localTTOp, const TensorTrain<float>& tt_x, const TensorTrain<float>& tt_b);
template bool check_localProblem(const TensorTrainOperator<double>& TTOpA, const TensorTrain<double>& TTx, const TensorTrain<double>& TTb, const TensorTrain<double>& TTw,
                                 bool ritzGalerkinProjection, SweepIndex swpIdx,
                                 const TensorTrainOperator<double>& localTTOp, const TensorTrain<double>& tt_x, const TensorTrain<double>& tt_b);
}