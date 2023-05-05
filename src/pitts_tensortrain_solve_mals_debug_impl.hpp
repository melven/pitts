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
#include "pitts_tensortrain_solve_mals_debug.hpp"
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
    }
  }
}


#endif // PITTS_TENSORTRAIN_SOLVE_MALS_DEBUG_IMPL_HPP
