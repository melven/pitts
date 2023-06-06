// only used for PITTS_DEVELOPMENT_BUILD
#ifndef PITTS_DEVELOP_BUILD
#error "pitts is a header-only library, .cpp files should only used internally to speedup pitts compile times"
#endif

// actually generate code for corresponding _impl.hpp file
#include "pitts_gmres_impl.hpp"

// additional includes
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_operator_apply_dense.hpp"
#include "pitts_multivector.hpp"
#include "pitts_multivector_axpby.hpp"
#include "pitts_multivector_dot.hpp"
#include "pitts_multivector_norm.hpp"
#include "pitts_multivector_scale.hpp"

using namespace PITTS;

template std::pair<Eigen::ArrayXd,Eigen::ArrayXd> PITTS::GMRES<Eigen::ArrayXd, TensorTrainOperator<double>, MultiVector<double>>(const TensorTrainOperator<double>&, bool, const MultiVector<double>&, MultiVector<double>&, int, const Eigen::ArrayXd&, const Eigen::ArrayXd&, const std::string_view&, bool);
template std::pair<Eigen::ArrayXf,Eigen::ArrayXf> PITTS::GMRES<Eigen::ArrayXf, TensorTrainOperator<float>, MultiVector<float>>(const TensorTrainOperator<float>&, bool, const MultiVector<float>&, MultiVector<float>&, int, const Eigen::ArrayXf&, const Eigen::ArrayXf&, const std::string_view&, bool);

template std::pair<Eigen::ArrayXd,Eigen::ArrayXd> PITTS::GMRES<Eigen::ArrayXd, TTOpApplyDenseHelper<double>, MultiVector<double>>(const TTOpApplyDenseHelper<double>&, bool, const MultiVector<double>&, MultiVector<double>&, int, const Eigen::ArrayXd&, const Eigen::ArrayXd&, const std::string_view&, bool);
template std::pair<Eigen::ArrayXf,Eigen::ArrayXf> PITTS::GMRES<Eigen::ArrayXf, TTOpApplyDenseHelper<float>, MultiVector<float>>(const TTOpApplyDenseHelper<float>&, bool, const MultiVector<float>&, MultiVector<float>&, int, const Eigen::ArrayXf&, const Eigen::ArrayXf&, const std::string_view&, bool);