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
#include "pitts_tensor2_qb_decomposition.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_normalize.hpp"

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
  //! @return               norm of the the resulting tensor TTy
  //!
  template<typename T>
  T axpby(T alpha, const TensorTrain<T>& TTx, T beta, TensorTrain<T>& TTy, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()))
  {
    // handle corner cases
    if( std::abs(alpha) == 0 )
      return beta;

    if( std::abs(beta) == 0 )
    {
      // TTy = TTx;
      TTy.editableSubTensors() = TTx.subTensors();
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
    // Subsequent orthogonalization algorithm based on svqb...
    //

    //double wtime = omp_get_wtime();
    double flops = 0;

    // Auxiliary tensor of rank-3, for adjusting the size of the subtensors!
    Tensor3<T> t3_tmp;

    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2_M(2,2);
    t2_M(0,0) = alpha*alpha;
    t2_M(0,1) = t2_M(1,0) = alpha*beta;
    t2_M(1,1) = beta*beta;
    Tensor2<T> t2_B(1,2);
    t2_B(0,0) = alpha;
    t2_B(0,1) = beta;
    int new_r1 = 1;

    Tensor2<T> last_t2_M;
    Tensor2<T> last_t2_B, t2_Binv;
    assert(TTx.subTensors().size() == TTy.subTensors().size());
    for(int iSubTensor = 0; iSubTensor < TTx.subTensors().size(); iSubTensor++)
    {
      const auto& subTx = TTx.subTensors()[iSubTensor];
      auto& subTy = TTy.editableSubTensors()[iSubTensor];
      const auto r1x = subTx.r1();
      const auto r2x = subTx.r2();
      const auto r1y = subTy.r1();
      const auto r2y = subTy.r2();
      assert(subTx.n() == subTy.n());
      const auto n = subTx.n();
      const auto r1sum = r1x+r1y;
      const bool lastSubTensor = iSubTensor+1 == TTx.subTensors().size();
      const auto r2sum = lastSubTensor ? 1 : r2x+r2y;
      const auto r2y_offset = lastSubTensor ? 0 : r2x;
      assert(subTx.nChunks() == subTy.nChunks());
      const auto nChunks = subTx.nChunks();

      flops += 2.*r1sum*r1sum*r2sum*r2sum*(n+1.);

      // copy last result
      std::swap(last_t2_M, t2_M);

      // prepare new result tensor for adding up
      t2_M.resize(r2sum,r2sum);
      for(int j = 0; j < r2sum; j++)
        for(int i = 0; i < r2sum; i++)
          t2_M(i,j) = T(0);
      T* t2data = &t2_M(0,0);
      const auto t2size = r2sum*r2sum;

      // prepare sub-tensor for replacing with possibly smaller version
      std::swap(subTy,t3_tmp);
      subTy.resize(new_r1,n,r2sum);
      // calculate
      // subTy = t2_B * (subTx 0;0 subTy)
      // t2_M = subTy^T subTy
#pragma omp parallel for schedule(static) reduction(+:t2data[:t2size])
      for(int k = 0; k < nChunks; k++)
      {
        for(int j = 0; j < r2sum; j++)
          for(int i = 0; i < new_r1; i++)
            subTy.chunk(i,k,j) = Chunk<T>{};
        // subTx part
        for(int j = 0; j < r2x; j++)
        {
          Chunk<T> row[r1x];
          for(int i = 0; i < r1x; i++)
            row[i] = subTx.chunk(i,k,j);
          for(int i_ = 0; i_ < r1x; i_++)
          {
            for(int i = 0; i < new_r1; i++)
              fmadd(t2_B(i,i_),row[i_],subTy.chunk(i,k,j));
          }
        }
        // subTy part
        for(int j = 0; j < r2y; j++)
        {
          Chunk<T> row[r1y];
          for(int i = 0; i < r1y; i++)
            row[i] = t3_tmp.chunk(i,k,j);
          for(int i_ = 0; i_ < r1y; i_++)
          {
            for(int i = 0; i < new_r1; i++)
              fmadd(t2_B(i,i_+r1x),row[i_],subTy.chunk(i,k,r2y_offset+j));
          }
        }
        // subTy^T subTy  part
        for(int j = 0; j < r2sum; j++)
        {
          // only consider upper triangular part (exploit symmetry)
          for(int i = 0; i <= j; i++)
          {
            Chunk<T> tmp{};
            for(int i_ = 0; i_ < new_r1; i_++)
              fmadd(subTy.chunk(i_,k,i), subTy.chunk(i_,k,j), tmp);
            // this directly works on a pointer for the data of t2_M to allow an OpenMP array reduction
            t2data[i+j*r2sum] += sum(tmp);
          }
        }
      }

      // copy upper triangular part to lower triangular part (symmetric!)
      for(int j = 0; j < r2sum; j++)
        for(int i = j+1; i < r2sum; i++)
          t2_M(i,j) = t2_M(j,i);

      // calculate t2_B with t2_B^T t2_B = t2_M
      std::swap(last_t2_B, t2_B);
      auto new_r2 = qb_decomposition(t2_M, t2_B, t2_Binv, rankTolerance);
      // handle zero case
      if( new_r2 == 0 )
      {
        TTy.setZero();
        return T(0);
      }

      // prepare sub-tensor for replacing with possibly smaller version
      std::swap(subTy,t3_tmp);
      subTy.resize(new_r1,n,new_r2);
      // calculate subT = subT * t2_Binv
#pragma omp parallel for schedule(static)
      for(int k = 0; k < nChunks; k++)
      {
        for(int i = 0; i < new_r1; i++)
        {
          Chunk<T> col[r2sum];
          for(int j = 0; j < r2sum; j++)
            col[j] = t3_tmp.chunk(i,k,j);
          for(int j = 0; j < new_r2; j++)
            subTy.chunk(i,k,j) = Chunk<T>{};
          for(int j_ = 0; j_ < r2sum; j_++)
          {
            for(int j = 0; j < new_r2; j++)
              fmadd(t2_Binv(j_,j), col[j_], subTy.chunk(i,k,j));
          }
        }
      }

      new_r1 = new_r2;
    }
    //wtime = omp_get_wtime()-wtime;
    //std::cout << "GFlop/s: " << flops/wtime*1.e-9 << std::endl;
    return t2_B(0,0) * normalize(TTy, rankTolerance);
  }

}


#endif // PITTS_TENSORTRAIN_AXPBY_HPP
