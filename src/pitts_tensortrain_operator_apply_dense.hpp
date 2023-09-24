/*! @file pitts_tensortrain_operator_apply_dense.hpp
* @brief apply a tensor train operator to a dense tensor
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2022-05-12
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_APPLY_DENSE_HPP
#define PITTS_TENSORTRAIN_OPERATOR_APPLY_DENSE_HPP

// includes
#include "pitts_tensortrain_operator.hpp"
#include "pitts_multivector.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Multiply a tensor train operator with a dense tensor
  //!
  //! Calculate y <- A * x
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TTOp           tensor train operator
  //! @param TTx            dense input tensor
  //! @param TTy            dense output tensor
  //!
  template<typename T>
  void apply(const TensorTrainOperator<T>& TTOp, const MultiVector<T>& MVx, MultiVector<T>& MVy);


  //! Helper type for continously applying the same tensor-train operator to dense tensors
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  class TTOpApplyDenseHelper
  {
    public:
      //! construct helper type from given tensor-train operator (copying and reshaping its sub-tensors)
      explicit TTOpApplyDenseHelper(const TensorTrainOperator<T>& TTOp);


      //! add padding to tensor-train so this operator can be applied to it
      void addPadding(MultiVector<T>& x) const;

      //! remove padding from a tensor-train 
      void removePadding(MultiVector<T>& y) const;

      //! (internal) preparation for addPadding: sets padding chunks to zero
      void preparePadding(MultiVector<T>& v) const;


      //! get the i-th sub-tensor (appropriately reshaped)
      const MultiVector<T>& A(int i) const {return A_.at(i);}

      //! get the i-th tensor train operator rank
      const int& rA(int i) const {return rA_.at(i);}

      //! get the (i+1)-th tensor train operator row dimension
      const int& r(int i) const {return r_.at(i);}

      //! get number of dimensions
      int nDim() const {return nDim_;}

      //! get the i-th (unpadded) dimension
      long long dim(int i) const {return dims_.at(i);}

      //! get the i-th padded dimension
      long long paddedDim(int i) const {return paddedDims_.at(i);}

      //! get total (unpadded) size of the lhs/rhs tensors
      long long nTotal() const {return nTotal_;}

      //! get total padded size of the lhs/rhs tensors
      long long nTotalPadded() const {return nTotalPadded_;}

      //! get the i-th buffer array
      MultiVector<T>& tmpv(int i) const {return tmpv_.at(i);}

    private:
      //! reshaped sub-tensors
      std::vector<MultiVector<T>> A_;

      //! dimension information (TTOp ranks)
      std::vector<int> rA_;

      //! dimension information (TTOp dims)
      std::vector<int> r_;

      //! number of dimensions
      int nDim_;

      //! original dimensions (unpadded)
      std::vector<long long> dims_;

      //! padded dimensions
      std::vector<long long> paddedDims_;

      //! total (unpadded) size of the lhs/rhs tensors
      long long nTotal_;

      //! total padded size of the lhs/rhs tensors
      long long nTotalPadded_;

      //! temporary buffers, stored for reusing without reallocating memory
      mutable std::vector<MultiVector<T>> tmpv_;
  };

  //! Faster multiplication of a tensor train operator with a dense tensor
  //!
  //! Calculate y <- A * x
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  //! @param TTOp           tensor train operator (helper type)
  //! @param TTx            dense input tensor in correct (padded) format (non-const to avoid internal copy, original restored on return)
  //! @param TTy            dense output tensor in correct (padded) format
  //!
  template<typename T>
  void apply(const TTOpApplyDenseHelper<T>& TTOp, MultiVector<T>& MVx, MultiVector<T>& MVy);

}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_operator_apply_dense_impl.hpp"
#endif

#endif // PITTS_TENSORTRAIN_OPERATOR_APPLY_DENSE_HPP
