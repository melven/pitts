// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_normalize.hpp
* @brief orthogonalization for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-17
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_NORMALIZE_HPP
#define PITTS_TENSORTRAIN_NORMALIZE_HPP

// includes
#include <limits>
#include <cmath>
#include "pitts_tensor2.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor2 and Tensor3 : A(:,*) * B(*,:,:)
    template<typename T>
    void normalize_contract1(const Tensor2<T>& A, const Tensor3<T>& B, Tensor3<T>& C);
    

    //! contract Tensor3 and Tensor2 : A(:,:,*) * B(*,:)
    template<typename T>
    void normalize_contract2(const Tensor3<T>& A, const Tensor2<T>& B, Tensor3<T>& C);
    

    //! scale operation for a Tensor3
    template<typename T>
    void t3_scale(T alpha, Tensor3<T>& x);
    

    //! Make a subset of sub-tensors left-orthogonal sweeping the given index range (left to right)
    //!
    //! @tparam T  underlying data type (double, complex, ...)
    //!
    //! @param TT             tensor in tensor train format
    //! @param firstIdx       index of the first sub-tensor to orthogonalize (0 <= firstIdx <= lastIdx)
    //! @param lastIdx        index of the last sub-tensor to orthogonalize (firstIdx <= lastIdx < nDim)
    //! @param rankTolerance  approximation tolerance
    //! @param maxRank        maximal allowed TT-rank, enforced even if this violates the rankTolerance
    //!
    template<typename T>
    void leftNormalize_range(TensorTrain<T>& TT, int firstIdx, int lastIdx, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max());
    

    //! Make a subset of sub-tensors right-orthogonal sweeping the given index range (right to left)
    //!
    //! @tparam T  underlying data type (double, complex, ...)
    //!
    //! @param TT             tensor in tensor train format
    //! @param firstIdx       index of the first sub-tensor to orthogonalize (0 <= firstIdx <= lastIdx)
    //! @param lastIdx        index of the last sub-tensor to orthogonalize (firstIdx <= lastIdx < nDim)
    //! @param rankTolerance  approximation tolerance
    //! @param maxRank        maximal allowed TT-rank, enforced even if this violates the rankTolerance
    //!
    template<typename T>
    void rightNormalize_range(TensorTrain<T>& TT, int firstIdx, int lastIdx, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max());
    

    //! Ensure a subset of sub-tensors is left-orthogonal and call leftNormalize_range if it is not.
    //!
    //! @tparam T  underlying data type (double, complex, ...)
    //!
    //! @param TT             tensor in tensor train format
    //! @param firstIdx       index of the first sub-tensor to orthogonalize (0 <= firstIdx <= lastIdx)
    //! @param lastIdx        index of the last sub-tensor to orthogonalize (firstIdx <= lastIdx < nDim)
    //!
    template<typename T>
    void ensureLeftOrtho_range(TensorTrain<T>& TT, int firstIdx, int lastIdx);
    

    //! Ensure a subset of sub-tensors is right-orthogonal and call rightNormalize_range if it is not.
    //!
    //! @tparam T  underlying data type (double, complex, ...)
    //!
    //! @param TT             tensor in tensor train format
    //! @param firstIdx       index of the first sub-tensor to orthogonalize (0 <= firstIdx <= lastIdx)
    //! @param lastIdx        index of the last sub-tensor to orthogonalize (firstIdx <= lastIdx < nDim)
    //!
    template<typename T>
    void ensureRightOrtho_range(TensorTrain<T>& TT, int firstIdx, int lastIdx);
    
  }

  //! TT-rounding: truncate tensor train by two normalization sweeps (first right to left, then left to right or vice-versa)
  //!
  //! Omits the first sweep if the tensor-train is already left- or right-orthogonal.
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TT             tensor in tensor train format, left-normalized on output
  //! @param rankTolerance  approximation tolerance
  //! @param maxRank        maximal allowed TT-rank, enforced even if this violates the rankTolerance
  //! @return               norm of the tensor
  //!
  template<typename T>
  T normalize(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max());
  

  //! Make all sub-tensors orthogonal sweeping left to right
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TT             tensor in tensor train format
  //! @param rankTolerance  approximation tolerance
  //! @param maxRank        maximal allowed TT-rank, enforced even if this violates the rankTolerance
  //! @return               norm of the tensor
  //!
  template<typename T>
  T leftNormalize(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max());
  

  //! Make all sub-tensors orthogonal sweeping right to left
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TT             tensor in tensor train format
  //! @param rankTolerance  approximation tolerance
  //! @param maxRank        maximal allowed TT-rank, enforced even if this violates the rankTolerance
  //! @return               norm of the tensor
  //!
  template<typename T>
  T rightNormalize(TensorTrain<T>& TT, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max());

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_normalize_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_NORMALIZE_HPP
