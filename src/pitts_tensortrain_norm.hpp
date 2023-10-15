// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensortrain_norm.hpp
* @brief norms for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-09
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_NORM_HPP
#define PITTS_TENSORTRAIN_NORM_HPP

// includes
#include "pitts_tensor2.hpp"
#include "pitts_tensor3.hpp"
#include "pitts_tensortrain.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! contract Tensor3 and Tensor2 along last dimensions: A(:,:,*) * B(:,*)
    template<typename T>
    void norm2_contract1(const Tensor3<T>& A, const Tensor2<T>& B, Tensor3<T>& C);
    

    //! contract Tensor3 and Tensor3 along the last two dimensions: A(:,*,*) * B(:,*,*)
    //!
    //! exploits the symmetry of the result in the norm calculation
    //!
    template<typename T>
    void norm2_contract2(const Tensor3<T>& A, const Tensor3<T>& B, Tensor2<T>& C);
    

    //! 2-norm of a Tensor3 contracting Tensor3 along all dimensions sqrt(A(*,*,*) * A(*,*,*))
    template<typename T>
    T t3_nrm(const Tensor3<T>& A);
    
  }

  //! calculate the 2-norm for a vector in tensor train format
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  T norm2(const TensorTrain<T>& TT);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_norm_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_NORM_HPP
