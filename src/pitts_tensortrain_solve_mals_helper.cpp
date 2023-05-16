// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_tensortrain_solve_mals_helper_impl.hpp"

namespace PITTS::internal::solve_mals
{
template void update_right_Ax(const TensorTrainOperator<float> TTOpA, const TensorTrain<float>& TTx, int firstIdx, int lastIdx,
                              std::vector<Tensor3<float>>& right_Ax, std::vector<Tensor3<float>>& right_Ax_ortho, std::vector<Tensor2<float>>& right_Ax_ortho_B);
template void update_right_Ax(const TensorTrainOperator<double> TTOpA, const TensorTrain<double>& TTx, int firstIdx, int lastIdx,
                              std::vector<Tensor3<double>>& right_Ax, std::vector<Tensor3<double>>& right_Ax_ortho, std::vector<Tensor2<double>>& right_Ax_ortho_B);

template void update_left_Ax(const TensorTrainOperator<float>& TTOpA, const TensorTrain<float>& TTx, int firstIdx, int lastIdx,
                             std::vector<Tensor3<float>>& left_Ax, std::vector<Tensor3<float>>& left_Ax_ortho, std::vector<Tensor2<float>>& left_Ax_ortho_B);
template void update_left_Ax(const TensorTrainOperator<double>& TTOpA, const TensorTrain<double>& TTx, int firstIdx, int lastIdx,
                             std::vector<Tensor3<double>>& left_Ax, std::vector<Tensor3<double>>& left_Ax_ortho, std::vector<Tensor2<double>>& left_Ax_ortho_B);

template TensorTrainOperator<float> setupProjectionOperator(const TensorTrain<float>& TTx, SweepIndex swpIdx);
template TensorTrainOperator<double> setupProjectionOperator(const TensorTrain<double>& TTx, SweepIndex swpIdx);

template TensorTrain<float> calculatePetrovGalerkinProjection(TensorTrainOperator<float>& TTAv, SweepIndex swpIdx, const TensorTrain<float>& TTx, bool symmetrize);
template TensorTrain<double> calculatePetrovGalerkinProjection(TensorTrainOperator<double>& TTAv, SweepIndex swpIdx, const TensorTrain<double>& TTx, bool symmetrize);

template void update_right_vTw(const RightPartialTT<float>& TTv, const RightPartialTT<float>& TTw, int firstIdx, int lastIdx, std::vector<Tensor2<float>>& right_vTw);
template void update_right_vTw(const RightPartialTT<double>& TTv, const RightPartialTT<double>& TTw, int firstIdx, int lastIdx, std::vector<Tensor2<double>>& right_vTw);

template void update_left_vTw(const LeftPartialTT<float>& TTv, const LeftPartialTT<float>& TTw, int firstIdx, int lastIdx, std::vector<Tensor2<float>>& left_vTw);
template void update_left_vTw(const LeftPartialTT<double>& TTv, const LeftPartialTT<double>& TTw, int firstIdx, int lastIdx, std::vector<Tensor2<double>>& left_vTw);

template TensorTrain<float> calculate_local_rhs(int iDim, int nMALS, const Tensor2<float>& left_vTb, const TensorTrain<float>& TTb, const Tensor2<float>& right_vTb);
template TensorTrain<double> calculate_local_rhs(int iDim, int nMALS, const Tensor2<double>& left_vTb, const TensorTrain<double>& TTb, const Tensor2<double>& right_vTb);
      
template TensorTrain<float> calculate_local_x(int iDim, int nMALS, const TensorTrain<float>& TTx);
template TensorTrain<double> calculate_local_x(int iDim, int nMALS, const TensorTrain<double>& TTx);
      
template TensorTrainOperator<float> calculate_local_op(int iDim, int nMALS, const Tensor2<float>& left_vTAx, const TensorTrainOperator<float>& TTOp, const Tensor2<float>& right_vTAx);
template TensorTrainOperator<double> calculate_local_op(int iDim, int nMALS, const Tensor2<double>& left_vTAx, const TensorTrainOperator<double>& TTOp, const Tensor2<double>& right_vTAx);
      
template float solveDenseGMRES(const TensorTrainOperator<float>& tt_OpA, bool symmetric, const TensorTrain<float>& tt_b, TensorTrain<float>& tt_x, int maxRank, int maxIter, float absTol, float relTol, const std::string& outputPrefix = "", bool verbose = false);
template double solveDenseGMRES(const TensorTrainOperator<double>& tt_OpA, bool symmetric, const TensorTrain<double>& tt_b, TensorTrain<double>& tt_x, int maxRank, int maxIter, double absTol, double relTol, const std::string& outputPrefix = "", bool verbose = false);
}