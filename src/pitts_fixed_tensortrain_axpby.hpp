/*! @file pitts_fixed_tensortrain_axpby.hpp
* @brief addition for simple tensor train format with fixed dimensions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-01-02
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_FIXED_TENSORTRAIN_AXPBY_HPP
#define PITTS_FIXED_TENSORTRAIN_AXPBY_HPP

// includes
#include <limits>
#include "pitts_tensor2.hpp"
#include "pitts_tensor2_qb_decomposition.hpp"
#include "pitts_fixed_tensortrain.hpp"
#include "pitts_fixed_tensor3_combine.hpp"
#include "pitts_fixed_tensor3_split.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Scale and add one tensor train to another
  //!
  //! Calculate gamma*y <- alpha*x + beta*y
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param alpha          scalar value, coefficient of TTx
  //! @param TTx            first tensor in tensor train format, must be leftNormalized
  //! @param beta           scalar value, coefficient of TTy
  //! @param TTy            second tensor in tensor train format, must be leftNormalized, overwritten with the result on output (still normalized!)
  //!
  template<typename T, int N>
  void axpby(T alpha, const FixedTensorTrain<T, N>& TTx, T beta, FixedTensorTrain<T, N>& TTy)
  {
    const auto timer = PITTS::timing::createScopedTimer<FixedTensorTrain<T,N>>();

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
    // Subsequent normalization...
    //

    // Auxiliary tensor of rank-3, for adjusting the size of the subtensors!
    FixedTensor3<T, N> t3_tmp;

    assert(TTx.nDims() == TTy.nDims());
    const auto nDims = TTx.nDims();
    for(int iDim = 0; iDim < nDims; iDim++)
    {
      const auto& subTx = TTx.subTensors()[iDim];
      auto& subTy = TTy.editableSubTensors()[iDim];
      const auto rx1 = subTx.r1();
      const auto rx2 = subTx.r2();
      const auto ry1 = subTy.r1();
      const auto ry2 = subTy.r2();

      const auto new_r1 = iDim == 0 ? 1 : rx1+ry1;
      const auto new_r2 = iDim == nDims-1 ? 1 : rx2+ry2;
      t3_tmp.resize(new_r1, new_r2);
      t3_tmp.setConstant(T{});

      // copy x to top left part...
      T effAlpha = iDim == nDims-1 ? alpha : T(1);
      for(int k = 0; k < rx2; k++)
        for(int j = 0; j < N; j++)
          for(int i = 0; i < rx1; i++)
            t3_tmp(i,j,k) = effAlpha * subTx(i,j,k);

      // copy y to bottom right part
      T effBeta = iDim == nDims-1 ? beta : T(1);
      const auto off1 = iDim == 0 ? 0 : rx1;
      const auto off2 = iDim == nDims-1 ? 0 : rx2;
      for(int k = 0; k < ry2; k++)
        for(int j = 0; j < N; j++)
          for(int i = 0; i < ry1; i++)
            t3_tmp(i+off1,j,k+off2) += effBeta * subTy(i,j,k);

      std::swap(subTy, t3_tmp);
    }

    // re-normalize with stupid combine/split...
    for(int iDim = 0; iDim+1 < nDims; iDim++)
    {
      auto& subT1 = TTy.editableSubTensors()[iDim];
      auto& subT2 = TTy.editableSubTensors()[iDim+1];
      const auto subT12 = combine(subT1, subT2);
      split(subT12, subT1, subT2);
    }
  }

}


#endif // PITTS_FIXED_TENSORTRAIN_AXPBY_HPP
