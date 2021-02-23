/*! @file pitts_tensortrain_axpby.hpp
* @brief addition for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-11-07
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_AXPBY_HPP
#define PITTS_TENSORTRAIN_AXPBY_HPP

// includes
//#include <omp.h>
//#include <iostream>
#include <cmath>
#include <limits>
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Scale and add one tensor train to another
  //!
  //! Calculate gamma*y <- alpha*x + beta*y
  //!
  //! @warning Both tensors must be leftNormalized.
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param alpha          scalar value, coefficient of TTx
  //! @param TTx            first tensor in tensor train format, must be leftNormalized
  //! @param beta           scalar value, coefficient of TTy
  //! @param TTy            second tensor in tensor train format, must be leftNormalized, overwritten with the result on output (still normalized!)
  //! @param rankTolerance  Approximation accuracy, used to reduce the TTranks of the result
  //! @param maxRank        maximal allowed TT-rank, enforced even if this violates the rankTolerance
  //! @return               norm of the the resulting tensor TTy
  //!
  template<typename T>
  T axpby(T alpha, const TensorTrain<T>& TTx, T beta, TensorTrain<T>& TTy, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max())
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // check that dimensions match
    if( TTx.dimensions() != TTy.dimensions() )
      throw std::invalid_argument("TensorTrain axpby dimension mismatch!");

    // handle corner cases
    if( std::abs(alpha) == 0 )
      return beta;

    if( std::abs(beta) == 0 )
    {
      // TTy = TTx;
      copy(TTx, TTy);
      return alpha;
    }

    // To add two tensor trains, for each sub-tensor, one obtains:
    //
    // a - a - ... - a - a
    // |   |         |   |
    //          +
    // b - b - ... - b - b
    // |   |         |   |
    //          =
    // axb - axb - ... - axb - axb
    //  |     |           |     |
    //
    // with axb := (a 0;
    //              0 b)
    //
    // With a subsequent orthogonalization step that tries to exploit the special structure of the matrices...
    //

    // Auxiliary tensor of rank-3
    Tensor3<T> t3_tmp;

    // Auxiliary tensor of rank-2
    Tensor2<T> t2_M(1,2);
    t2_M(0,0) = alpha;
    t2_M(0,1) = beta;

    const int nDim = TTx.subTensors().size();
    for(int iDim = 0; iDim < nDim; iDim++)
    {
      const auto& subTx = TTx.subTensors()[iDim];
      auto& subTy = TTy.editableSubTensors()[iDim];
      const auto r1x = subTx.r1();
      const auto r2x = subTx.r2();
      const auto r1y = subTy.r1();
      const auto r2y = subTy.r2();
      const auto n = subTx.n();
      const bool lastSubTensor = iDim+1 == TTx.subTensors().size();
      const auto r2sum = lastSubTensor ? 1 : r2x+r2y;

      // first contract t2_M with (TTx 0; 0 TTy)
      const auto r1_new = t2_M.r1();
      assert(t2_M.r2() == r1x+r1y);
      t3_tmp.resize(r1_new,n,r2sum);
      for(int i = 0; i < r1_new; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2x; k++)
          {
            T tmp{};
            for(int l = 0; l < r1x; l++)
              tmp += t2_M(i,l) * subTx(l,j,k);
            t3_tmp(i,j,k) = tmp;
          }
      for(int i = 0; i < r1_new; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2y; k++)
          {
            T tmp{};
            for(int l = 0; l < r1y; l++)
              tmp += t2_M(i,r1x+l) * subTy(l,j,k);
            if( lastSubTensor )
              t3_tmp(i,j,k) += tmp;
            else
              t3_tmp(i,j,r2x+k) = tmp;
          }

      if( lastSubTensor )
      {
        // no need for any further steps, we do a normalize afterwards anyway!
        std::swap(subTy, t3_tmp);
        break;
      }

      // now calculate SVD of t3_tmp(: : x :)
      t2_M.resize(r1_new*n, r2sum);
      for(int i = 0; i < r1_new; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2sum; k++)
            t2_M(i+j*r1_new,k) = t3_tmp(i,j,k);

      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      //Eigen::BDCSVD<EigenMatrix> svd(ConstEigenMap(t2_M), Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::JacobiSVD<EigenMatrix> svd(ConstEigenMap(t2_M), Eigen::ComputeThinU | Eigen::ComputeThinV);
      svd.setThreshold(rankTolerance);
      const auto r2_new = svd.rank();

      // we always need at least rank 1
      if( r2_new == 0 )
      {
        subTy.resize(r1_new, n, 1);
        for(int i = 0; i < r1_new; i++)
          for(int j = 0; j < n; j++)
            subTy(i,j,0) = T(0);

        t2_M.resize(1,r2sum);
        for(int i = 0; i < r2sum; i++)
          t2_M(0,i) = T(0);
      }
      else // r2_new > 0
      {
        subTy.resize(r1_new, n, r2_new);
        for(int i = 0; i < r1_new; i++)
          for(int j = 0; j < n; j++)
            for(int k = 0; k < r2_new; k++)
              subTy(i,j,k) = svd.matrixU()(i+j*r1_new,k);

        t2_M.resize(r2_new,r2sum);
        EigenMap(t2_M) = svd.singularValues().topRows(r2_new).asDiagonal() * svd.matrixV().leftCols(r2_new).adjoint();
      }
    }

    return normalize(TTy, rankTolerance, maxRank);
  }

}


#endif // PITTS_TENSORTRAIN_AXPBY_HPP
