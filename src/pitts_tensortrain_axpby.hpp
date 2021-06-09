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
#include <cassert>
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract block Tensor3 and Tensor2: (X 0; 0 Y)(:,:,*) * B(*,:)
    template<typename T>
    void axpby_contract1(const Tensor3<T>& X, const Tensor3<T>& Y, const Tensor2<T>& B, Tensor3<T>& C)
    {
      const auto r1x = X.r1();
      const auto r1y = Y.r1();
      const auto r1sum = r1x + r1y;
      const auto n = X.n();
      assert(X.n() == Y.n());
      const auto r2x = X.r2();
      const auto r2y = Y.r2();
      const auto r2sum = r2x + r2y;
      assert(B.r1() == r2sum);
      const auto r2new = B.r2();
      C.resize(r1sum, n, r2new);


      //const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
      //  {{"r1", "nChunks", "r2", "r2_"},{r1, nChunks, r2, r2_}}, // arguments
      //  {{r1*nChunks*r2*r2_*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
      //   {(r1*nChunks*r2+r2*r2_)*kernel_info::Load<Chunk<T>>() + (r1*nChunks*r2_)*kernel_info::Store<Chunk<T>>()}} // data transfers
      //  );


      for(int i = 0; i < r1x; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2new; k++)
          {
            T tmp{};
            for(int l = 0; l < r2x; l++)
              tmp += X(i,j,l) * B(l,k);
            C(i,j,k) = tmp;
          }
      for(int i = 0; i < r1y; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2new; k++)
          {
            T tmp{};
            for(int l = 0; l < r2y; l++)
              tmp += Y(i,j,l) * B(r2x+l,k);
            C(r1x+i,j,k) = tmp;
          }

    }

  }


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
    Tensor2<T> t2_M(2,1);
    t2_M(0,0) = alpha;
    t2_M(1,0) = beta;

    const int nDim = TTx.subTensors().size();
    for(int iDim = nDim-1; iDim >= 0; iDim--)
    {
      const auto& subTx = TTx.subTensors()[iDim];
      auto& subTy = TTy.editableSubTensors()[iDim];

      internal::axpby_contract1(subTx, subTy, t2_M, t3_tmp);

      const auto r1 = t3_tmp.r1();
      const auto n = t3_tmp.n();
      const auto r2 = t3_tmp.r2();

      if( iDim == 0 )
      {
        // no need for any further steps, we do a normalize afterwards anyway!
        assert(r1 == 2);
        // combine result (first subTensor is (TTx TTy) instead of (TTx 0; 0 TTy)
        subTy.resize(1, n, r2);
        for(int i = 0; i < n; i++)
          for(int j = 0; j < r2; j++)
            subTy(0, i, j) = t3_tmp(0, i, j) + t3_tmp(1, i, j);
        break;
      }


      // now calculate SVD of t3_tmp(: x : :)
      t2_M.resize(r1, n*r2);
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2; k++)
            t2_M(i, j*r2+k) = t3_tmp(i,j,k);

      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      Eigen::BDCSVD<EigenMatrix> svd(ConstEigenMap(t2_M), Eigen::ComputeThinU | Eigen::ComputeThinV);
      svd.setThreshold(rankTolerance);
      const auto r1new = svd.rank();

      // we always need at least rank 1
      if( r1new == 0 )
      {
        subTy.resize(1, n, r2);
        subTy.setConstant(T(0));

        t2_M.resize(r1,1);
        for(int i = 0; i < r1; i++)
          t2_M(i,0) = T(0);
      }
      else // r1new > 0
      {
        subTy.resize(r1new, n, r2);
        for(int i = 0; i < r1new; i++)
          for(int j = 0; j < n; j++)
            for(int k = 0; k < r2; k++)
              subTy(i,j,k) = svd.matrixV().leftCols(r1new).adjoint()(i,j*r2+k);

        t2_M.resize(r1,r1new);
        EigenMap(t2_M) = svd.matrixU().leftCols(r1new) * svd.singularValues().topRows(r1new).asDiagonal();
      }
    }

    return leftNormalize(TTy, rankTolerance, maxRank);
  }

}


#endif // PITTS_TENSORTRAIN_AXPBY_HPP
