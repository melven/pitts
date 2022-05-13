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
    void axpby_contract1(const Tensor3<T>& X, const Tensor3<T>& Y, const Tensor2<T>& B, Tensor3<T>& C, bool first)
    {
      const auto r1x = X.r1();
      const auto r1y = Y.r1();
      // special case, for first==true, we calculate (X Y)(:,:,*) * B(*,:)
      const auto r1sum = first ? r1x : r1x + r1y;
      const auto n = X.n();
      const auto nChunks = X.nChunks();
      assert(X.n() == Y.n());
      const auto r2x = X.r2();
      const auto r2y = Y.r2();
      const auto r2sum = r2x + r2y;
      assert(B.r1() == r2sum);
      const auto r2new = B.r2();
      C.resize(r1sum, n, r2new);


      const auto timer = PITTS::performance::createScopedTimer<TensorTrain<T>>(
        {{"r1sum", "nChunks", "r2new"},{r1sum, nChunks, r2new}}, // arguments
        {{(r1x*nChunks*r2x*r2new + r1y*nChunks*r2y*r2new)*Chunk<T>::size*kernel_info::FMA<T>()}, // flops
         {(r1x*nChunks*r2x + r1y*nChunks*r2y + r2sum*r2new)*kernel_info::Load<Chunk<T>>() + (r1sum*nChunks*r2new)*kernel_info::Store<Chunk<T>>()}} // data transfers
        );


#pragma omp parallel
{
#pragma omp for collapse(3) schedule(static) nowait
      for(int i = 0; i < r1x; i++)
        for(int jChunk = 0; jChunk < nChunks; jChunk++)
          for(int k = 0; k < r2new; k++)
          {
            Chunk<T> tmp{};
            for(int l = 0; l < r2x; l++)
              fmadd(B(l,k), X.chunk(i,jChunk,l), tmp);
            C.chunk(i,jChunk,k) = tmp;
          }
if( !first )
{
#pragma omp for collapse(3) schedule(static) nowait
      for(int i = 0; i < r1y; i++)
        for(int jChunk = 0; jChunk < nChunks; jChunk++)
          for(int k = 0; k < r2new; k++)
          {
            Chunk<T> tmp{};
            for(int l = 0; l < r2y; l++)
              fmadd(B(r2x+l,k), Y.chunk(i,jChunk,l), tmp);
            C.chunk(r1x+i,jChunk,k) = tmp;
          }
}
else
{
#pragma omp for collapse(3) schedule(static) nowait
      for(int i = 0; i < r1y; i++)
        for(int jChunk = 0; jChunk < nChunks; jChunk++)
          for(int k = 0; k < r2new; k++)
          {
            Chunk<T> tmp = C.chunk(i,jChunk,k);
            for(int l = 0; l < r2y; l++)
              fmadd(B(r2x+l,k), Y.chunk(i,jChunk,l), tmp);
            C.chunk(i,jChunk,k) = tmp;
          }
}
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

    const int nDim = TTx.subTensors().size();
    if( TTx.subTensors()[0].r1() != TTy.subTensors()[0].r1() || TTx.subTensors()[nDim-1].r2() != TTy.subTensors()[nDim-1].r2() )
      throw std::invalid_argument("TensorTrain axpby boundary ranks mismatch!");

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
    Tensor2<T> t2_M;
    {
      const int r2 = TTx.subTensors()[nDim-1].r2();
      t2_M.resize(2*r2, r2);
      for(int i = 0; i < r2; i++)
        for(int j = 0; j < r2; j++)
        {
          t2_M(i,j) = i == j ? alpha : T(0);
          t2_M(r2+i,j) = i == j ? beta : T(0);
        }
    }

    for(int iDim = nDim-1; iDim >= 0; iDim--)
    {
      const auto& subTx = TTx.subTensors()[iDim];
      auto& subTy = TTy.editableSubTensors()[iDim];

      internal::axpby_contract1(subTx, subTy, t2_M, t3_tmp, iDim == 0);

      const auto r1 = t3_tmp.r1();
      const auto n = t3_tmp.n();
      const auto nChunks = t3_tmp.nChunks();
      const auto r2 = t3_tmp.r2();

      if( iDim == 0 )
      {
        // no need for any further steps, we do a normalize afterwards anyway!
        copy(t3_tmp, subTy);
        break;
      }


      // now calculate SVD of t3_tmp(: x : :)
      t2_M.resize(n*r2, r1);
      for(int k = 0; k < r2; k++)
        for(int i = 0; i < r1; i++)
          for(int j = 0; j < n; j++)
            t2_M(j+k*n,i) = t3_tmp(i,j,k);

      const auto [Q,B] = internal::normalize_qb(t2_M);
      const auto r1new = Q.cols();

      subTy.resize(r1new, n, r2);
      for(int k = 0; k < r2; k++)
        for(int i = 0; i < r1new; i++)
          for(int j = 0; j < n; j++)
            subTy(i,j,k) = Q(j+k*n,i);

      t2_M.resize(r1,r1new);
      EigenMap(t2_M) = B.transpose();
    }

    return leftNormalize(TTy, rankTolerance, maxRank);
  }

}


#endif // PITTS_TENSORTRAIN_AXPBY_HPP
