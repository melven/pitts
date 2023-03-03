/*! @file pitts_tensortrain_solve_gmres.hpp
* @brief TT-GMRES algorithm, iterative solver for linear systems in tensor-train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-07-04
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_SOLVE_GMRES_HPP
#define PITTS_TENSORTRAIN_SOLVE_GMRES_HPP

// includes
#include <vector>
#include <limits>
#include <string_view>
#include <string>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_operator.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! helper function for converting an array to a string
    template<typename T>
    std::string to_string(const std::vector<T>& v);
  }

  //! TT-GMRES: iterative solver for linear systems in tensor-train format
  //!
  //! Given A, b, calculates x with Ax=b approximately up to a given tolerance.
  //!
  //! @tparam T             data type (double, float, complex)
  //!
  //! @param TTOpA              tensor-train operator A
  //! @param TTb                right-hand side tensor-train b
  //! @param TTx                initial guess on input, overwritten with the (approximate) result on output
  //! @param maxIter            maximal number of iterations
  //! @param absResTol          absolute residual tolerance: the iteration aborts if the absolute residual norm is smaller than absResTol
  //! @param relResTol          relative residual tolerance: the iteration aborts if the relative residual norm is smaller than relResTol
  //! @param maxRank            maximal allowed TT-rank, enforced even if this violates the residualTolerance
  //! @param adaptiveTolerance  use an adaptive tolerance for the tensor-train arithmetic in the iteration
  //! @param symmetric          set to true for symmetric operators to exploit the symmetry (results in a MinRes variant)
  //! @param outputPrefix       string to prefix all output about the convergence history
  //! @param verbose            set to true to print the residual norm in each iteration to std::cout
  //! @return                   residual norm of the result (||Ax - b||)
  //!
  template <typename T>
  T solveGMRES(const TensorTrainOperator<T> &TTOpA, const TensorTrain<T> &TTb, TensorTrain<T> &TTx,
               int maxIter, T absResTol, T relResTol,
               int maxRank = std::numeric_limits<int>::max(), bool adaptiveTolerance = true, bool symmetric = false,
               const std::string_view &outputPrefix = "", bool verbose = false);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_solve_gmres_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_SOLVE_GMRES_HPP
