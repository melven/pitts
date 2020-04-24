/*! @file pitts_tensortrain_normalize.hpp
* @brief orthogonalization for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-17
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_NORMALIZE_HPP
#define PITTS_TENSORTRAIN_NORMALIZE_HPP

// includes
//#include <omp.h>
//#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_qb_decomposition.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  template<typename T>
  T normalize(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()))
  {
    const auto norm = rightNormalize(TT, rankTolerance);
    return norm * leftNormalize(TT, rankTolerance);
  }

  //! Make all sub-tensors orthogonal sweeping left to right
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TT   tensor in tensor train format
  //! @return     norm of the tensor
  //!
  template<typename T>
  T leftNormalize(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()))
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // Transforms the tensor train in the following invariant way
    //
    // o--o--   --o--o
    // |  |  ...  |  |
    //
    // becomes
    //
    // q--q--   --q--*
    // |  |  ...  |  |
    //
    // where the "q" are all left-orthogonal.
    //
    // Algorithm based on svqb...
    //

    //double wtime = omp_get_wtime();
    double flops = 0;

    // Auxiliary tensor of rank-3, for adjusting the size of the subtensors!
    Tensor3<T> t3_tmp;

    // Auxiliary tensor of rank-2, currently contracted
    Tensor2<T> t2_M(1,1);
    t2_M(0,0) = T(1);
    Tensor2<T> t2_B(1,1);
    t2_B(0,0) = T(1);
    int new_r1 = 1;

    Tensor2<T> last_t2_M;
    Tensor2<T> last_t2_B, t2_Binv;
    for(auto& subT: TT.editableSubTensors())
    {
      const auto r1 = subT.r1();
      const auto r2 = subT.r2();
      const auto n = subT.n();
      const auto nChunks = subT.nChunks();

      flops += 2.*r1*r1*r2*r2*(n+1.);

      // loop unrolling parameter (but unrolling is done by hand below!)
      constexpr auto unrollSize = 2;

      // copy last result
      const int r1_padded = unrollSize * (1 + (r1-1)/unrollSize);
      last_t2_M.resize(r1_padded,r1_padded);
      for(int j_ = 0; j_ < r1_padded; j_++)
        for(int i_ = 0; i_ < r1_padded; i_++)
          last_t2_M(i_,j_) = (i_ < r1 && j_ < r1) ? t2_M(i_,j_) : T(0);

      // prepare new result tensor for adding up
      t2_M.resize(r2,r2);
      for(int j = 0; j < r2; j++)
        for(int i = 0; i < r2; i++)
          t2_M(i,j) = T(0);
      T* t2data = &t2_M(0,0);
      const auto t2size = r2*r2;

      // prepare sub-tensor for replacing with possibly smaller version
      std::swap(subT,t3_tmp);
      subT.resize(new_r1,n,r2);
      // calculate
      // subT = t2_B * subT
      // t2_M = subT^T subT
#pragma omp parallel for schedule(static) reduction(+:t2data[:t2size])
      for(int k = 0; k < nChunks; k++)
      {
        for(int j = 0; j < r2; j++)
        {
          Chunk<T> row[r1];
          for(int i = 0; i < r1; i++)
            row[i] = t3_tmp.chunk(i,k,j);
          for(int i = 0; i < new_r1; i++)
            subT.chunk(i,k,j) = Chunk<T>{};
          for(int i_ = 0; i_ < r1; i_++)
          {
            for(int i = 0; i < new_r1; i++)
              fmadd(t2_B(i,i_),row[i_],subT.chunk(i,k,j));
          }
          // unly consider upper triangular part (exploit symmetry)
          for(int i = 0; i <= j; i++)
          {
            Chunk<T> tmp{};
            for(int i_ = 0; i_ < new_r1; i_++)
              fmadd(subT.chunk(i_,k,i), subT.chunk(i_,k,j), tmp);
            // this directly works on a pointer for the data of t2_M to allow an OpenMP array reduction
            t2data[i+j*r2] += sum(tmp);
          }
        }
      }

      // copy upper triangular part to lower triangular part (symmetric!)
      for(int j = 0; j < r2; j++)
        for(int i = j+1; i < r2; i++)
          t2_M(i,j) = t2_M(j,i);

      // calculate t2_B with t2_B^T t2_B = t2_M
      std::swap(last_t2_B, t2_B);
      auto new_r2 = qb_decomposition(t2_M, t2_B, t2_Binv, rankTolerance);
      // handle zero case
      if( new_r2 == 0 )
      {
        TT.setZero();
        return T(0);
      }

      // prepare sub-tensor for replacing with possibly smaller version
      std::swap(subT,t3_tmp);
      subT.resize(new_r1,n,new_r2);
      // calculate subT = subT * t2_Binv
#pragma omp parallel for schedule(static)
      for(int k = 0; k < nChunks; k++)
      {
        for(int i = 0; i < new_r1; i++)
        {
          Chunk<T> col[r2];
          for(int j = 0; j < r2; j++)
            col[j] = t3_tmp.chunk(i,k,j);
          for(int j = 0; j < new_r2; j++)
            subT.chunk(i,k,j) = Chunk<T>{};
          for(int j_ = 0; j_ < r2; j_++)
          {
            for(int j = 0; j < new_r2; j++)
              fmadd(t2_Binv(j_,j), col[j_], subT.chunk(i,k,j));
          }
        }
      }

      new_r1 = new_r2;
    }
    //wtime = omp_get_wtime()-wtime;
    //std::cout << "GFlop/s: " << flops/wtime*1.e-9 << std::endl;
    return t2_B(0,0);
  }


  template<typename T>
  T rightNormalize(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()))
  {
    const auto timer = PITTS::timing::createScopedTimer<TensorTrain<T>>();

    // transpose and leftNormalize for stupidity for now
    auto reverseDims = TT.dimensions();
    std::reverse(reverseDims.begin(), reverseDims.end());
    TensorTrain<T> tmpTT(reverseDims);
    const auto nDim = TT.dimensions().size();
    for(int iDim = 0; iDim < nDim; iDim++)
    {
      auto& subT = tmpTT.editableSubTensors()[nDim-1-iDim];
      const auto& oldSubT = TT.subTensors()[iDim];
      const auto r1 = oldSubT.r2();
      const auto n = oldSubT.n();
      const auto r2 = oldSubT.r1();
      subT.resize(r1,n,r2);
      for(int j = 0; j < r2; j++)
        for(int k = 0; k < n; k++)
          for(int i = 0; i < r1; i++)
            subT(i,k,j) = oldSubT(j,k,i);
    }
    const auto norm = leftNormalize(tmpTT, rankTolerance);
    for(int iDim = 0; iDim < nDim; iDim++)
    {
      auto& subT = TT.editableSubTensors()[nDim-1-iDim];
      const auto& oldSubT = tmpTT.subTensors()[iDim];
      const auto r1 = oldSubT.r2();
      const auto n = oldSubT.n();
      const auto r2 = oldSubT.r1();
      subT.resize(r1,n,r2);
      for(int j = 0; j < r2; j++)
        for(int k = 0; k < n; k++)
          for(int i = 0; i < r1; i++)
            subT(i,k,j) = oldSubT(j,k,i);
    }
    return norm;
  }


}


#endif // PITTS_TENSORTRAIN_NORMALIZE_HPP
