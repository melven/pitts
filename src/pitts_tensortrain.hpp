/*! @file pitts_tensortrain.hpp
* @brief simple tensor train format
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSORTRAIN_HPP
#define PITTS_TENSORTRAIN_HPP

// includes
#include <vector>
#include <exception>
#include "pitts_tensor3.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! tensor train class
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  class TensorTrain
  {
    public:
      //! tensor dimensions
      //!
      //! These are constant as one usually only changes the ranks of the individual sub-tensors in the tensor train.
      //!
      const auto& dimensions() const {return dimensions_;}

      //! allow const access to all sub-tensors
      const auto& subTensors() const {return subTensors_;}

      //! allow non-const access to all sub-tensors
      //!
      //! \warning Do not modify sub-tensor dimensions here, only their values!
      //!
      auto& editableSubTensors() {return subTensors_;}

      //! create a new tensor train that represents a n^d tensor
      TensorTrain(int d, int n, int initial_TTrank = 1) : TensorTrain(std::vector<int>(d,n), initial_TTrank) {}

      //! create a new tensor with the given dimensions
      TensorTrain(const std::vector<int>& dimensions, int initial_TTrank = 1)
        : dimensions_(dimensions)
      {
        // create all sub-tensors
        subTensors_.resize(dimensions.size());
        setTTranks(initial_TTrank);
      }

      //! explicit copy construction is ok
      TensorTrain(const TensorTrain<T>& other)
        : dimensions_(other.dimensions_)
      {
        // create and copy all subtensors
        subTensors_.resize(dimensions_.size());
        for(int i = 0; i < dimensions_.size(); i++)
          copy(other.subTensors()[i], subTensors_[i]);
      }

      //! no implicit copy assignment
      const TensorTrain<T>& operator=(const TensorTrain<T>&) = delete;

      //! move construction is ok
      TensorTrain(TensorTrain<T>&&) = default;

      //! move assignment is ok
      TensorTrain<T>& operator=(TensorTrain<T>&&) = default;


      //! set sub-tensor dimensions (TT-ranks), destroying all existing data
      void setTTranks(const std::vector<int>& tt_ranks)
      {
        if( tt_ranks.size() != dimensions_.size() - 1 )
          throw std::invalid_argument("TensorTrain: wrong number of TTranks!");
        // set first and last individually
        subTensors_.front().resize(1, dimensions_.front(), tt_ranks.front());
        for(int i = 1; i < dimensions_.size()-1; i++)
          subTensors_[i].resize(tt_ranks[i-1], dimensions_[i], tt_ranks[i]);
        subTensors_.back().resize(tt_ranks.back(), dimensions_.back(), 1);
      }

      //! set sub-tensor dimensions (TT-ranks), destroying all existing data
      void setTTranks(int tt_rank)
      {
        subTensors_.front().resize(1, dimensions_.front(), tt_rank);
        for(int i = 1; i < dimensions_.size()-1; i++)
          subTensors_[i].resize(tt_rank, dimensions_[i], tt_rank);
        subTensors_.back().resize(tt_rank, dimensions_.back(), 1);
      }

      //! get current sub-tensor dimensions (TT-ranks)
      std::vector<int> getTTranks() const
      {
        std::vector<int> tt_ranks(dimensions_.size() - 1);
        for(int i = 0; i < dimensions_.size() - 1; i++)
          tt_ranks[i] = subTensors_[i].r2();
        return tt_ranks;
      }

      //! make this a tensor of zeros
      //!
      //! \warning intended for testing purposes
      //!
      void setZero()
      {
        // we use a rank of one...
        setTTranks(1);
        for(auto& M: subTensors_)
          M.setConstant(T(0));
      }

      //! make this a tensor of ones
      //!
      //! \warning intended for testing purposes
      //!
      void setOnes()
      {
        // we use a rank of one...
        setTTranks(1);
        for(auto& M: subTensors_)
          M.setConstant(T(1));
      }

      //! make this a canonical unit tensor in the given direction
      //!
      //! \warning intended for testing purposes
      //!
      void setUnit(const std::vector<int>& index)
      {
        if( index.size() != dimensions_.size() )
          throw std::invalid_argument("TensorTrain: multidimensional index array has wrong dimension!");
        // we use a rank of one...
        setTTranks(1);
        for(int i = 0; i < dimensions_.size(); i++)
          subTensors_[i].setUnit(0, index[i], 0);
      }

    private:
      //! tensor dimensions
      //!
      //! These are constant as one usually only changes the ranks of the individual sub-tensors in the tensor train.
      //!
      std::vector<int> dimensions_;

      //! actual data: array of sub-tensors
      //!
      //! resulting tensor train is defined as the tensor network:
      //!
      //! o -- o -- ... -- o -- o
      //! |    |           |    |
      //!
      //! where each "o" denotes a tensor of rank 3, respectively rank 2 on the boundaries
      //!
      std::vector<Tensor3<T>> subTensors_;
  };

  //! explicitly copy a TensorTrain object
  template<typename T>
  void copy(const TensorTrain<T>& a, TensorTrain<T>& b)
  {
    // check that dimensions match
    if( a.dimensions() != b.dimensions() )
      throw std::invalid_argument("TensorTrain copy dimension mismatch!");

    for(int i = 0; i < a.dimensions().size(); i++)
      copy(a.subTensors()[i], b.editableSubTensors()[i]);
  }
}


#endif // PITTS_TENSORTRAIN_HPP
