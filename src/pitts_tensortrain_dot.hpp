/*! @file pitts_tensortrain_dot.hpp
* @brief inner products for simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-10
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_DOT_HPP
#define PITTS_TENSORTRAIN_DOT_HPP

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
    void dot_contract1(const Tensor3<T>& A, const Tensor2<T>& B, Tensor3<T>& C);
    

    //! contract Tensor3 and Tensor2 along last and first dimensions: A(:,:,*) * B(*,:)
    template<typename T>
    void dot_contract1t(const Tensor3<T>& A, const Tensor2<T>& B, Tensor3<T>& C);
    

    //! contract Tensor3 and Tensor2 along first dimensions: A(*,:) * B(*,:,:)
    template<typename T>
    void reverse_dot_contract1(const Tensor2<T>& A, const Tensor3<T>& B, Tensor3<T>& C);
    

    //! contract Tensor3 and Tensor3 along the last two dimensions: A(:,*,*) * B(:,*,*)
    template<typename T>
    void dot_contract2(const Tensor3<T>& A, const Tensor3<T>& B, Tensor2<T>& C);
    

    //! contract Tensor3 and Tensor3 along the first two dimensions: A(*,*,:) * B(*,*,:)
    template<typename T>
    void reverse_dot_contract2(const Tensor3<T>& A, const Tensor3<T>& B, Tensor2<T>& C);
    

    //! contract Tensor3 and Tensor3 along all dimensions: A(*,*,*) * B(*,*,*)
    template<typename T>
    T t3_dot(const Tensor3<T>& A, const Tensor3<T>& B);
    
  }

  //! calculate the inner product for two vectors in tensor train format
  //!
  //! This also allows to calculate the dot product of two tensor trains with boundary ranks > 1
  //! (e.g. from extracting a sub-tensor-train from the middle of another tensor train)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  T dot(const TensorTrain<T>& TT1, const TensorTrain<T>& TT2);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_dot_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_DOT_HPP
