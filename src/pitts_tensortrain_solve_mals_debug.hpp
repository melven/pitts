/*! @file pitts_tensortrain_solve_mals_debug.hpp
* @brief Error checking functionality for PITTS::solveMALS
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2023-05-05
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SOLVE_MALS_DEBUG_HPP
#define PITTS_TENSORTRAIN_SOLVE_MALS_DEBUG_HPP

// includes
#include <cmath>
#include <limits>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_operator.hpp"
#include "pitts_tensortrain_sweep_index.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    
    //! dedicated helper functions for solveMALS
    namespace solve_mals
    {
      //! helper function: return TensorTrain with additional dimension instead of boundary rank
      template<typename T>
      TensorTrain<T> removeBoundaryRank(const TensorTrain<T>& tt);

      //! helper function: return TensorTrainTrain without additional empty (1x1) dimension
      template<typename T>
      TensorTrain<T> removeBoundaryRankOne(const TensorTrainOperator<T>& ttOp);
      
      //! subtract two Tensor3 for checking differences...
      template<typename T>
      Tensor3<T> operator-(const Tensor3<T>& a, const Tensor3<T>& b);

      //! check that left/right_Ax = TTOpA * TTx
      template<typename T>
      bool check_Ax(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx, SweepIndex swpIdx, const std::vector<Tensor3<T>>& Ax);

      //! check that v^T v = I and A v x_local = Av and
      template<typename T>
      bool check_ProjectionOperator(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx, SweepIndex swpIdx, const TensorTrainOperator<T>& TTv, const TensorTrainOperator<T>& TTAv);

      //! check w^Tw = I
      template<typename T>
      bool check_Orthogonality(SweepIndex swpIdx, const TensorTrain<T>& TTw);

      //! check that dimensions of A, x and b fit to define A*x=b
      template<typename T>
      bool check_systemDimensions(const TensorTrainOperator<T>& localTTOp, const TensorTrain<T>& tt_x, const TensorTrain<T>& tt_b);

      //! check that the local problem is correct
      template<typename T>
      bool check_localProblem(const TensorTrainOperator<T>& TTOpA, const TensorTrain<T>& TTx, const TensorTrain<T>& TTb, const TensorTrain<T>& TTw,
                              bool ritzGalerkinProjection, SweepIndex swpIdx,
                              const TensorTrainOperator<T>& localTTOp, const TensorTrain<T>& tt_x, const TensorTrain<T>& tt_b);
    }
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_solve_mals_debug_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_SOLVE_MALS_DEBUG_HPP
