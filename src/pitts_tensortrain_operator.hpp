/*! @file pitts_tensortrain_operator.hpp
* @brief simple operator in tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2021-02-11
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_OPERATOR_HPP
#define PITTS_TENSORTRAIN_OPERATOR_HPP

// includes
#include <algorithm>
#include <cassert>
#include "pitts_tensortrain.hpp"
#include "pitts_tensortrain_axpby.hpp"
#include "pitts_tensortrain_random.hpp"
#include "pitts_tensortrain_normalize.hpp"
#include "pitts_performance.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    //! helper function to return the element-wise product of two arrays
    inline auto dimension_product(const std::vector<int>& row_dimensions, const std::vector<int>& column_dimensions)
    {
      if( row_dimensions.size() != column_dimensions.size() )
        throw std::invalid_argument("The number of row dimensions and column dimensions doesn't match!");

      // helper variable for the product of cols*rows per dimension
      std::vector<int> totalDims;
      std::transform(row_dimensions.begin(), row_dimensions.end(),
          column_dimensions.begin(),
          std::back_inserter(totalDims),
          std::multiplies<>{});
      return totalDims;
    }
  }

  //! tensor train operator class
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  class TensorTrainOperator
  {
    public:
      //! output tensor dimensions of this operator (like rows of a matrix)
      //!
      //! These are constant as one usually only changes the ranks of the individual sub-tensors in the tensor train.
      //!
      const auto& row_dimensions() const {return row_dimensions_;}

      //! input tensor dimension of this operator (like columns of a matrix)
      //!
      //! These are constant as one usually only changes the ranks of the individual sub-tensors in the tensor train.
      //!
      const auto& column_dimensions() const {return column_dimensions_;}

      //! allow const access to the underlying tensor train data structure (low-level access)
      const auto& tensorTrain() const {return tensorTrain_;}

      //! allow non-const access to the underlying tensor train data structure (low-level access)
      auto& tensorTrain() {return tensorTrain_;}

      //! create a new tensor train operator that represents a m^d -> n^d operator
      TensorTrainOperator(int d, int n, int m, int initial_TTrank = 1) : TensorTrainOperator(std::vector<int>(d,n), std::vector<int>(d,m), initial_TTrank) {}

      //! create a new tensor operator with the given dimensions
      TensorTrainOperator(const std::vector<int>& row_dimensions,
                          const std::vector<int>& column_dimensions,
                          int initial_TTrank = 1)
        : row_dimensions_(row_dimensions),
          column_dimensions_(column_dimensions),
          tensorTrain_(internal::dimension_product(row_dimensions, column_dimensions), initial_TTrank)
      {
      }

      //! explicit copy construction is ok
      TensorTrainOperator(const TensorTrainOperator<T>& other) = default;

      //! no implicit copy assignment
      const TensorTrainOperator<T>& operator=(const TensorTrainOperator<T>&) = delete;

      //! move construction is ok
      TensorTrainOperator(TensorTrainOperator<T>&&) = default;

      //! move assignment is ok
      TensorTrainOperator<T>& operator=(TensorTrainOperator<T>&&) = default;


      //! set sub-tensor dimensions (TT-ranks), destroying all existing data
      void setTTranks(const std::vector<int>& tt_ranks)
      {
        tensorTrain_.setTTranks(tt_ranks);
      }

      //! set sub-tensor dimensions (TT-ranks), destroying all existing data
      void setTTranks(int tt_rank)
      {
        tensorTrain_.setTTranks(tt_rank);
      }

      //! get current sub-tensor dimensions (TT-ranks)
      std::vector<int> getTTranks() const
      {
        return tensorTrain_.getTTranks();
      }

      //! make this a tensor of zeros
      void setZero()
      {
        tensorTrain_.setZero();
      }

      //! make this a tensor of ones
      void setOnes()
      {
        tensorTrain_.setOnes();
      }

      //! make this an identity operator
      void setEye()
      {
        tensorTrain_.setTTranks(1);
        Tensor3<T> newSubT;
        for(int iDim = 0; iDim < row_dimensions_.size(); iDim++)
        {
          newSubT.resize(1, tensorTrain_.dimensions()[iDim], 1);
          newSubT.setConstant(T(0));
          for(int i = 0; i < std::min(row_dimensions_[iDim], column_dimensions_[iDim]); i++)
            newSubT(0,index(iDim,i,i),0) = T(1);
          newSubT = tensorTrain_.setSubTensor(iDim, std::move(newSubT));
        }
      }

      // helper function to calculate the index in the 3d-tensor of the tensor train
      auto index(int iDim, int i, int j) const
      {
        return i + j*row_dimensions_[iDim];
      }

    private:
      //! output tensor dimensions of this operator (like rows of a matrix)
      //!
      //! These are constant as one usually only changes the ranks of the individual sub-tensors in the tensor train.
      //!
      std::vector<int> row_dimensions_;

      //! input tensor dimension of this operator (like columns of a matrix)
      //!
      //! These are constant as one usually only changes the ranks of the individual sub-tensors in the tensor train.
      //!
      std::vector<int> column_dimensions_;

      //! actual data: stored in a TensorTrain, we just reindex it here
      TensorTrain<T> tensorTrain_;
  };


  //! explicitly copy a TensorTrain object
  template<typename T>
  void copy(const TensorTrainOperator<T>& a, TensorTrainOperator<T>& b)
  {
    // check that dimensions match
    if( a.row_dimensions() != b.row_dimensions() )
      throw std::invalid_argument("TensorTrainOperator copy row dimension mismatch!");
    if( a.column_dimensions() != b.column_dimensions() )
      throw std::invalid_argument("TensorTrainOperator copy column dimension mismatch!");

    copy(a.tensorTrain(), b.tensorTrain());
  }


  //! Scale and add one tensor train operator to another
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
  void axpby(T alpha, const TensorTrainOperator<T>& TTOpx, T beta, TensorTrainOperator<T>& TTOpy, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()))
  {
    // check that dimensions match
    if( TTOpx.row_dimensions() != TTOpy.row_dimensions() )
      throw std::invalid_argument("TensorTrainOperator axpby row dimension mismatch!");
    if( TTOpx.column_dimensions() != TTOpy.column_dimensions() )
      throw std::invalid_argument("TensorTrainOperator axpby column dimension mismatch!");

    const auto gamma = axpby(alpha, TTOpx.tensorTrain(), beta, TTOpy.tensorTrain(), rankTolerance);
    const int nDim = TTOpy.tensorTrain().dimensions().size();
    if( nDim > 0 )
    {
      Tensor3<T> newSubT;
      copy(TTOpy.tensorTrain().subTensor(nDim-1), newSubT);
      internal::t3_scale(gamma, newSubT);
      TTOpy.tensorTrain().setSubTensor(nDim-1, std::move(newSubT));
    }
  }


  //! fill a tensor train operator with random values (keeping current TT-ranks)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  void randomize(TensorTrainOperator<T>& TTOp)
  {
    randomize(TTOp.tensorTrain());
  }


  //! normalize a tensor train operator (reducing its' ranks if possible)
  //!
  //! @tparam T underlying data type (double, complex, ...)
  //!
  template<typename T>
  T normalize(TensorTrainOperator<T>& TTOp, T rankTolerance = std::sqrt(std::numeric_limits<T>::epsilon()), int maxRank = std::numeric_limits<int>::max())
  {
    return normalize(TTOp.tensorTrain(), rankTolerance, maxRank);
  }
}


#endif // PITTS_TENSORTRAIN_OPERATOR_HPP
