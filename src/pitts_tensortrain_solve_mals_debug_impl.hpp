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
        const auto sqrt_eps = std::sqrt(std::numeric_limits<T>::epsilon());

        TensorTrain<T> TTAx_err(TTOpA.row_dimensions());
        apply(TTOpA, TTx, TTAx_err);
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
        assert(WtW_err < 100*sqrt_eps);

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
