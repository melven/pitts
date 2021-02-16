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
#include "pitts_tensor2_eigen_adaptor.hpp"
#include "pitts_tensortrain.hpp"
#include "pitts_timer.hpp"
#include "pitts_chunk_ops.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! TT-rounding: truncate tensor train by two normalization sweeps (first right to left, then left to right)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TT             tensor in tensor train format, left-normalized on output
  //! @param rankTolerance  approximation tolerance
  //! @return               norm of the tensor
  //!
  template<typename T>
  T normalize(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()))
  {
    const auto norm = rightNormalize(TT, T(0));
    return norm * leftNormalize(TT, rankTolerance);
  }

  //! Make all sub-tensors orthogonal sweeping left to right
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TT             tensor in tensor train format
  //! @param rankTolerance  approximation tolerance
  //! @return               norm of the tensor
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

    // matrix from previous step
    Tensor2<T> t2_M(1,1);
    t2_M(0,0) = T(1);

    // auxiliary tensor of rank-3
    Tensor3<T> t3_tmp;

    for(auto& subT: TT.editableSubTensors())
    {
      const auto r1 = subT.r1();
      const auto r2 = subT.r2();
      const auto n = subT.n();

      const auto r1_new = t2_M.r1();
      assert(r1 == t2_M.r2());

      // first multiply subT with t2_M
      t3_tmp.resize(r1_new, n, r2);
      for(int i = 0; i < r1_new; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2; k++)
          {
            T tmp{};
            for(int l = 0; l < r1; l++)
              tmp += t2_M(i,l) * subT(l,j,k);
            t3_tmp(i,j,k) = tmp;
          }

      // now calculate SVD of t3_tmp(: : x :)
      t2_M.resize(r1_new*n, r2);
      for(int i = 0; i < r1_new; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2; k++)
            t2_M(i+j*r1_new,k) = t3_tmp(i,j,k);

      using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      Eigen::BDCSVD<EigenMatrix> svd(ConstEigenMap(t2_M), Eigen::ComputeThinU | Eigen::ComputeThinV);
      svd.setThreshold(rankTolerance);
      const auto r2_new = svd.rank();

      subT.resize(r1_new, n, r2_new);
      for(int i = 0; i < r1_new; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2_new; k++)
            subT(i,j,k) = svd.matrixU()(i+j*r1_new,k);

      t2_M.resize(r2_new,r2);
      EigenMap(t2_M) = svd.singularValues().topRows(r2_new).asDiagonal() * svd.matrixV().leftCols(r2_new).adjoint();
    }
    return t2_M(0,0);
  }


  //! Make all sub-tensors orthogonal sweeping right to left
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TT             tensor in tensor train format
  //! @param rankTolerance  approximation tolerance
  //! @return               norm of the tensor
  //!
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
