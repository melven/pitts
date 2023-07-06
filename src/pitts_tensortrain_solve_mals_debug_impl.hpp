/*! @file pitts_tensortrain_solve_mals_debug_impl.hpp
* @brief Error checking functionality for PITTS::solveMALS
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-05-05
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SOLVE_MALS_DEBUG_IMPL_HPP
#define PITTS_TENSORTRAIN_SOLVE_MALS_DEBUG_IMPL_HPP

// includes
#include <cassert>
#include <vector>
#include <cmath>
#include "pitts_tensortrain_solve_mals_debug.hpp"
#include "pitts_tensortrain_solve_mals_helper.hpp"
#include "pitts_tensortrain_debug.hpp"
#include "pitts_tensortrain_operator_debug.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    
    //! dedicated helper functions for solveMALS
    namespace solve_mals
    {
      // helper function: return TensorTrain with additional dimension instead of boundary rank
      template<typename T>
      TensorTrain<T> removeBoundaryRank(const TensorTrain<T>& tt)
      {
        const int nDim = tt.dimensions().size();
        std::vector<Tensor3<T>> subTensors(nDim);
        for(int iDim = 0; iDim < nDim; iDim++)
          copy(tt.subTensor(iDim), subTensors[iDim]);
        const int rleft = subTensors.front().r1();
        const int rright = subTensors.back().r2();
        if( rleft != 1 )
        {
          // generate identity
          Tensor3<T> subT(1, rleft, rleft);
          subT.setConstant(T(0));
          for(int i = 0; i < rleft; i++)
            subT(0, i, i) = T(1);
          subTensors.insert(subTensors.begin(), std::move(subT));
        }
        if( rright != 1 )
        {
          // generate identity again
          Tensor3<T> subT(rright, rright, 1);
          subT.setConstant(T(0));
          for(int i = 0; i < rright; i++)
            subT(i, i, 0) = T(1);
          subTensors.insert(subTensors.end(), std::move(subT));
        }

        TensorTrain<T> extendedTT(std::move(subTensors));
        return extendedTT;
      }

      // helper function: return TensorTrainTrain without additional empty (1x1) dimension
      template<typename T>
      TensorTrain<T> removeBoundaryRankOne(const TensorTrainOperator<T>& ttOp)
      {
        std::vector<int> rowDims = ttOp.row_dimensions();
        std::vector<int> colDims = ttOp.column_dimensions();
        std::vector<Tensor3<T>> subTensors(rowDims.size());
        for(int iDim = 0; iDim < rowDims.size(); iDim++)
          copy(ttOp.tensorTrain().subTensor(iDim), subTensors[iDim]);
        
        if( rowDims.front() == 1 && colDims.front() == 1 && subTensors.front()(0,0,0) == T(1) )
        {
          rowDims.erase(rowDims.begin());
          colDims.erase(colDims.begin());
          subTensors.erase(subTensors.begin());
        }
        if( rowDims.back() == 1 && colDims.back() == 1 && subTensors.back()(0,0,0) == T(1) )
        {
          rowDims.pop_back();
          colDims.pop_back();
          subTensors.pop_back();
        }

        TensorTrain<T> ttOp_(std::move(subTensors));

        return ttOp_;
      }


      // subtract two Tensor3 for checking differences...
      template<typename T>
      Tensor3<T> operator-(const Tensor3<T>& a, const Tensor3<T>& b)
      {
        assert(a.r1() == b.r1());
        assert(a.n() == b.n());
        assert(a.r2() == b.r2());
        Tensor3<T> c(a.r1(), a.n(), a.r2());
        for(int i = 0; i < a.r1(); i++)
          for (int j = 0; j < a.n(); j++)
            for (int k = 0; k < a.r2(); k++)
              c(i,j,k) = a(i,j,k) - b(i,j,k);
        return c;
      }

      // check that left/right_Ax = TTOpA * TTx
      template<typename T>
      bool check_Ax(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx, SweepIndex swpIdx, const std::vector<Tensor3<T>>& Ax)
      {
        using PITTS::debug::operator*;

        const auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());

        const TensorTrain<T> Ax_ref = TTOpA * TTx;
        assert(Ax.size() == swpIdx.nDim());
        for(int iDim = 0; iDim < swpIdx.nDim(); iDim++)
        {
          if( iDim >= swpIdx.leftDim() && iDim <= swpIdx.rightDim() )
            continue;
          const T error = internal::t3_nrm(Ax_ref.subTensor(iDim) - Ax[iDim]);
          assert(error < sqrt_eps);
        }
        return true;
      }

      // check that left/right_Ax_ortho = left/rightNormalize(TTAx)
      template<typename T>
      bool check_Ax_ortho(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx, const std::vector<std::pair<Tensor3<T>,Tensor2<T>>>& Ax_ortho)
      {
        using PITTS::debug::operator*;

        const auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());

        TensorTrain<T> TTAx_err = TTOpA * TTx;
        const int nDim = TTAx_err.dimensions().size();
        assert(Ax_ortho.size() == nDim);
        std::vector<Tensor3<T>> tmpAx(nDim);
        for(int i = 0; i < nDim; i++)
          copy(Ax_ortho[i].first, tmpAx[i]);
        TensorTrain<T> TTAx(std::move(tmpAx));
        const T sgn = dot(TTAx, TTAx_err) > 0 ? T(-1) : T(1);
        const T error = axpby(sgn*norm2(TTAx_err), TTAx, T(1), TTAx_err, T(0));
        assert(error < sqrt_eps);
        return true;
      }

      // check that left/right_Ax_b_ortho = left/rightNormalize(TTAx-TTb)
      template<typename T>
      bool check_Ax_b_ortho(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx, const TensorTrain<T>& TTb, T alpha_Ax, T alpha_b, bool leftToRight, const std::vector<std::pair<Tensor3<T>,Tensor2<T>>>& Ax_b_ortho)
      {
        using PITTS::debug::operator*;
        using PITTS::debug::operator-;
        using mat = Eigen::MatrixX<T>;

        const auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());

        TensorTrain<T> TTAx = TTOpA * TTx;
        const T alpha_Ax_err = std::abs(alpha_Ax) - norm2(TTAx);
        assert(std::abs(alpha_Ax_err) < sqrt_eps);
        TensorTrain<T> TTAx_b_err = TTAx - TTb;
        const int nDim = TTAx_b_err.dimensions().size();
        assert(Ax_b_ortho.size() == nDim);
        std::vector<Tensor3<T>> tmpAx_b(nDim);
        for(int i = 0; i < nDim; i++)
          copy(Ax_b_ortho[i].first, tmpAx_b[i]);
        // first or last boundary rank can be 2 -> need to contract then
        if( !leftToRight )
        {
          mat Mleft(1,2);
          Mleft << alpha_Ax, -alpha_b;
          auto mapB = ConstEigenMap(Ax_b_ortho.front().second);
          const auto& subT = Ax_b_ortho.front().first;
          Eigen::Map<const mat> mapSubT(&subT(0,0,0), subT.r1(), subT.n()*subT.r2());
          Tensor3<T> newSubT(1, subT.n(), subT.r2());
          Eigen::Map<mat> mapNewSubT(&newSubT(0,0,0), 1, subT.n()*subT.r2());
          mat tmpB(Mleft.cols(), mapSubT.rows());
          tmpB << mat::Identity(tmpB.rows()-mapB.rows(), tmpB.cols()), mapB;
          mapNewSubT = Mleft * tmpB * mapSubT;
          tmpAx_b.front() = std::move(newSubT);
        }
        else
        {
          mat Mright(2,1);
          Mright << alpha_Ax,
                   -alpha_b;
          auto mapB = ConstEigenMap(Ax_b_ortho.back().second);
          const auto& subT = Ax_b_ortho.back().first;
          Eigen::Map<const mat> mapSubT(&subT(0,0,0), subT.r1()*subT.n(), subT.r2());
          Tensor3<T> newSubT(subT.r1(), subT.n(), 1);
          Eigen::Map<mat> mapNewSubT(&newSubT(0,0,0), subT.r1()*subT.n(), 1);
          mat tmpB(mapSubT.cols(), Mright.rows());
          tmpB << mat::Identity(tmpB.rows(), tmpB.cols()-mapB.cols()), mapB;
          mapNewSubT = mapSubT * tmpB * Mright;
          tmpAx_b.back() = std::move(newSubT);
        }
        TensorTrain<T> TTAx_b(std::move(tmpAx_b));
        const T sgn = dot(TTAx_b, TTAx_b_err) > 0 ? T(-1) : T(1);
        const T error = axpby(sgn, TTAx_b, T(1), TTAx_b_err, T(0));
        assert(error < sqrt_eps);
        return true;
      }

      // check that v^T v = I and A v x_local = Av and
      template<typename T>
      bool check_ProjectionOperator(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx, SweepIndex swpIdx, const TensorTrainOperator<T>& TTv, const TensorTrainOperator<T>& TTAv)
      {
        using namespace PITTS::debug;
        using PITTS::debug::operator*;
        using PITTS::debug::operator-;

        const auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());

        // check orthogonality
        TensorTrainOperator<T> TTOpI(TTv.column_dimensions(), TTv.column_dimensions());
        TTOpI.setEye();
        const TensorTrainOperator<T> vTv = transpose(TTv) * TTv;
        const T vTv_err_norm = norm2((vTv - TTOpI).tensorTrain());
        if( swpIdx.nMALS() == 1 )
        {
          if( vTv_err_norm >= 1000*sqrt_eps )
            std::cout << "vTv-I err: " << vTv_err_norm << std::endl;
          assert(vTv_err_norm < 1000*sqrt_eps);
        }
        else
        {
          assert(vTv_err_norm < sqrt_eps);
        }
        // check A V x_local = A x
        TensorTrain<T> tt_x = calculate_local_x(swpIdx.leftDim(), swpIdx.nMALS(), TTx);
        TensorTrain<T> TTXlocal(TTv.column_dimensions());
        TTXlocal.setOnes();
        const int leftOffset = ( tt_x.subTensor(0).r1() > 1 ) ? -1 :  0;
        TTXlocal.setSubTensors(swpIdx.leftDim() + leftOffset, removeBoundaryRank(tt_x));
        assert(norm2( TTOpA * TTx - TTAv * TTXlocal ) < sqrt_eps);

        return true;
      }

      // check w^Tw = I
      template<typename T>
      bool check_Orthogonality(SweepIndex swpIdx, const TensorTrain<T>& TTw)
      {
        using namespace PITTS::debug;
        using PITTS::debug::operator*;
        using PITTS::debug::operator-;

        const auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());

        // check orthogonality
        TensorTrainOperator<T> TTOpW = setupProjectionOperator(TTw, swpIdx);
        TensorTrainOperator<T> TTOpI(TTOpW.column_dimensions(), TTOpW.column_dimensions());
        TTOpI.setEye();
        const TensorTrainOperator<T> WtW = transpose(TTOpW) * TTOpW;
        const T WtW_err = norm2((WtW - TTOpI).tensorTrain());
        if( WtW_err > 100*sqrt_eps )
        {
          std::cout << "WtW_err: " << WtW_err << std::endl;
          return false;
        }

        return true;
      }

      // check AMEn subspace
      template<typename T>
      bool check_AMEnSubspace(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTv, const TensorTrain<T>& TTx, const TensorTrain<T>& TTb, SweepIndex swpIdx, bool leftToRight, const TensorTrain<T>& tt_z)
      {
        using namespace PITTS::debug;
        using PITTS::debug::operator*;
        using PITTS::debug::operator-;
        using mat = Eigen::MatrixX<T>;
        using vec = Eigen::VectorX<T>;

        TensorTrainOperator<T> TTOpW = setupProjectionOperator(TTv, swpIdx);
        if( leftToRight )
        {
          std::vector<int> colDimOpW_left(TTOpW.column_dimensions().size());
          for(int iDim = 0; iDim < swpIdx.nDim(); iDim++)
            colDimOpW_left[iDim] = iDim < swpIdx.rightDim() ? TTOpW.column_dimensions()[iDim] : TTOpW.row_dimensions()[iDim];
          TensorTrainOperator<T> TTOpW_left(TTOpW.row_dimensions(), colDimOpW_left);
          TTOpW_left.setEye();
          std::vector<Tensor3<T>> tmpSubT(swpIdx.leftDim());

          for(int iDim = 0; iDim < swpIdx.leftDim(); iDim++)
            copy(TTOpW.tensorTrain().subTensor(iDim), tmpSubT[iDim]);
          TTOpW_left.tensorTrain().setSubTensors(0, std::move(tmpSubT));
          TensorTrain<T> TTz = transpose(TTOpW_left) * (TTb - TTOpA * TTx);
          internal::leftNormalize_range(TTz, 0, swpIdx.rightDim(), T(0));
          internal::rightNormalize_range(TTz, swpIdx.rightDim(), swpIdx.nDim()-1, T(0));
          assert(swpIdx.nMALS() == 1);
          const int iDim = swpIdx.leftDim();

          const auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());
          
          Tensor2<T> z_ref;
          if( iDim != 0 )
          {
            // need to contract sub-tensors iDim-1 and iDim (boundary-rank vs. normal TT)
            const auto& subT_prev = TTz.subTensor(iDim-1);
            const auto& subT = TTz.subTensor(iDim);
            assert(subT_prev.r1() == 1);
            assert(subT_prev.n() == tt_z.subTensor(0).r1());
            assert(subT.n() == tt_z.subTensor(0).n());
            //assert(subT.r2() == tt_z.subTensor(0).r2());
            Eigen::Map<const mat> map_prev(&subT_prev(0,0,0), subT_prev.n(), subT_prev.r2());
            Eigen::Map<const mat> map(&subT(0,0,0), subT.r1(), subT.n()*subT.r2());
            z_ref.resize(map_prev.rows(), map.cols());
            EigenMap(z_ref) = map_prev * map;
            z_ref.resize(subT_prev.n()*subT.n(), subT.r2(), false);
          }
          const auto [Q, B] = internal::normalize_svd(unfold_left(tt_z.subTensor(0)), true, sqrt_eps);
          const auto [Q_ref, B_ref] = internal::normalize_svd(
            (iDim == 0 ? unfold_left(TTz.subTensor(iDim)) : z_ref), 
            true, sqrt_eps);

          Eigen::BDCSVD<mat> svd(ConstEigenMap(B)), svd_ref(ConstEigenMap(B_ref));
          const T sigma0 = svd_ref.singularValues()(0);
          vec sigma_err = vec::Zero(std::max(svd_ref.singularValues().size(), svd.singularValues().size()));
          sigma_err.topRows(svd.singularValues().size()) = svd.singularValues();
          sigma_err.topRows(svd_ref.singularValues().size()) -= svd_ref.singularValues();
          const int nmin = std::min(svd_ref.singularValues().size(), svd.singularValues().size());
          auto mapQ = ConstEigenMap(Q).leftCols(nmin);
          auto mapQ_ref = ConstEigenMap(Q_ref).leftCols(nmin);
          mat Q_err = (mapQ - mapQ_ref * mapQ_ref.transpose() * mapQ) * svd_ref.singularValues().topRows(nmin).asDiagonal();
          //std::cout << "Q_err:\n" << Q_err << "\n";
          //std::cout << "singular values:\n" << svd_ref.singularValues().transpose() << std::endl;
          //std::cout << "singular values error:\n" << sigma_err.transpose() << std::endl;
          const T Q_error = Q_err.array().abs().maxCoeff();
          assert(Q_error <= 10*sqrt_eps*std::max(sigma0, T(1)));
          const T sigma_error = sigma_err.array().abs().maxCoeff();
          assert(sigma_error <= 10*sqrt_eps*std::max(sigma0, T(1)));
        }
        else // !leftToRight
        {
          std::vector<int> colDimOpW_right(TTOpW.column_dimensions().size());
          for(int iDim = 0; iDim < swpIdx.nDim(); iDim++)
            colDimOpW_right[iDim] = iDim >= swpIdx.leftDim() ? TTOpW.column_dimensions()[iDim] : TTOpW.row_dimensions()[iDim];
          TensorTrainOperator<T> TTOpW_right(TTOpW.row_dimensions(), colDimOpW_right);
          TTOpW_right.setEye();
          std::vector<Tensor3<T>> tmpSubT(swpIdx.nDim()-swpIdx.leftDim());

          for(int iDim = 0; iDim < tmpSubT.size(); iDim++)
            copy(TTOpW.tensorTrain().subTensor(swpIdx.leftDim()+iDim), tmpSubT[iDim]);
          TTOpW_right.tensorTrain().setSubTensors(swpIdx.leftDim(), std::move(tmpSubT));
          TensorTrain<T> TTz = transpose(TTOpW_right) * (TTb - TTOpA * TTx);
          internal::leftNormalize_range(TTz, 0, swpIdx.leftDim(), T(0));
          internal::rightNormalize_range(TTz, swpIdx.leftDim(), swpIdx.nDim()-1, T(0));
          assert(swpIdx.nMALS() == 1);
          const int iDim = swpIdx.leftDim();

          const auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());

          Tensor2<T> z_ref;
          if( iDim != swpIdx.nDim()-1 )
          {
            // need to contract sub-tensors iDim+1 and iDim (boundary-rank vs. normal TT)
            const auto& subT = TTz.subTensor(iDim);
            const auto& subT_next = TTz.subTensor(iDim+1);
            assert(subT_next.r2() == 1);
            assert(subT_next.n() == tt_z.subTensor(0).r2());
            assert(subT.n() == tt_z.subTensor(0).n());
            Eigen::Map<const mat> map(&subT(0,0,0), subT.r1()*subT.n(), subT.r2());
            Eigen::Map<const mat> map_next(&subT_next(0,0,0), subT_next.r1(), subT_next.n());

            z_ref.resize(map.rows(), map_next.cols());
            EigenMap(z_ref) = map * map_next;
            z_ref.resize(subT.r1(), subT.n()*subT_next.n(), false);
          }
          const auto [B, Qt] = internal::normalize_svd(unfold_right(tt_z.subTensor(0)), false, sqrt_eps);
          const auto [B_ref, Qt_ref] = internal::normalize_svd(
            (iDim == swpIdx.nDim()-1 ? unfold_right(TTz.subTensor(iDim)) : z_ref), 
            false, sqrt_eps);

          Eigen::BDCSVD<mat> svd(ConstEigenMap(B)), svd_ref(ConstEigenMap(B_ref));
          const T sigma0 = svd_ref.singularValues()(0);
          vec sigma_err = vec::Zero(std::max(svd_ref.singularValues().size(), svd.singularValues().size()));
          sigma_err.topRows(svd.singularValues().size()) = svd.singularValues();
          sigma_err.topRows(svd_ref.singularValues().size()) -= svd_ref.singularValues();
          const int nmin = std::min(svd_ref.singularValues().size(), svd.singularValues().size());
          auto mapQ = ConstEigenMap(Qt).topRows(nmin).transpose();
          auto mapQ_ref = ConstEigenMap(Qt_ref).topRows(nmin).transpose();
          mat Q_err = (mapQ - mapQ_ref * mapQ_ref.transpose() * mapQ) * svd_ref.singularValues().topRows(nmin).asDiagonal();
          //std::cout << "Q_err:\n" << Q_err << "\n";
          //std::cout << "singular values:\n" << svd_ref.singularValues().transpose() << std::endl;
          //std::cout << "singular values error:\n" << sigma_err.transpose() << std::endl;
          const T Q_error = Q_err.array().abs().maxCoeff();
          assert(Q_error <= 10*sqrt_eps*std::max(sigma0, T(1)));
          const T sigma_error = sigma_err.array().abs().maxCoeff();
          assert(sigma_error <= 10*sqrt_eps*std::max(sigma0, T(1)));
        }

        return true;
      }

      // check that dimensions of localTTOp, tt_x and tt_b are ok (so localTTOp*tt_x - tt_b is valid)
      template<typename T>
      bool check_systemDimensions(const TensorTrainOperator<T>& localTTOp, const TensorTrain<T>& tt_x, const TensorTrain<T>& tt_b)
      {
        assert(tt_x.dimensions().size() == tt_b.dimensions().size());
        const int nMALS = tt_x.dimensions().size();
        assert(localTTOp.row_dimensions().size() == nMALS+2);
        assert(localTTOp.column_dimensions().size() == nMALS+2);
        // check that localTTOp * tt_x is valid
        assert(localTTOp.column_dimensions()[0] == tt_x.subTensor(0).r1());
        for(int i = 0; i < nMALS; i++)
          assert(localTTOp.column_dimensions()[i+1] == tt_x.dimensions()[i]);
        assert(localTTOp.column_dimensions()[nMALS+1] == tt_x.subTensor(nMALS-1).r2());
        // check that localTTOp^T * tt_b is valid
        assert(localTTOp.row_dimensions()[0] == tt_b.subTensor(0).r1());
        for(int i = 0; i < nMALS; i++)
          assert(localTTOp.row_dimensions()[i+1] == tt_b.dimensions()[i]);
        assert(localTTOp.row_dimensions()[nMALS+1] == tt_b.subTensor(nMALS-1).r2());

        return true;
      }

      // check that the local problem is correct
      template<typename T>
      bool check_localProblem(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx, const TensorTrain<T>& TTb, const TensorTrain<T>& TTw,
                              bool ritzGalerkinProjection, SweepIndex swpIdx,
                              const TensorTrainOperator<T>& localTTOp, const TensorTrain<T>& tt_x, const TensorTrain<T>& tt_b)
      {
        using namespace PITTS::debug;
        using PITTS::debug::operator*;
        using PITTS::debug::operator-;

        const auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());

        if( ritzGalerkinProjection )
        {
          assert(std::abs(dot(tt_x, tt_b) - dot(TTx, TTb)) <= sqrt_eps*std::abs(dot(TTx, TTb)));
          return true;
        }

        TensorTrainOperator<T> TTOpV = setupProjectionOperator(TTx, swpIdx);
        TensorTrainOperator<T> TTOpW = setupProjectionOperator(TTw, swpIdx);
        // check W^T b = tt_b
        TensorTrain<T> TTblocal(TTOpW.column_dimensions());
        TTblocal.setOnes();
        int leftOffset = ( tt_b.subTensor(0).r1() > 1 ) ? -1 :  0;
        TTblocal.setSubTensors(swpIdx.leftDim() + leftOffset, removeBoundaryRank(tt_b));
        assert(norm2( transpose(TTOpW) * TTb -  TTblocal ) < sqrt_eps);

        // check localTTOp = W^T A V
        TensorTrainOperator<T> WtAV_ref = transpose(TTOpW) * TTOpA * TTOpV;
        TensorTrainOperator<T> WtAV(TTOpW.column_dimensions(), TTOpV.column_dimensions());
        WtAV.setOnes();
        leftOffset = ( localTTOp.tensorTrain().subTensor(0).n() == 1 && localTTOp.tensorTrain().subTensor(0)(0,0,0) == T(1) ) ? 0 : -1;
        WtAV.tensorTrain().setSubTensors(swpIdx.leftDim() + leftOffset, removeBoundaryRankOne(localTTOp));
        assert(norm2( WtAV.tensorTrain() - WtAV_ref.tensorTrain() ) < sqrt_eps*norm2(WtAV_ref.tensorTrain()));

        return true;
      }
    }
  }
}


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_DEBUG_IMPL_HPP
