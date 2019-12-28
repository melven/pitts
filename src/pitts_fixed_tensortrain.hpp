/*! @file pitts_fixed_tensortrain.hpp
* @brief simple tensor train format with compile-time dimension
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-28
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_FIXED_TENSORTRAIN_HPP
#define PITTS_FIXED_TENSORTRAIN_HPP

// includes
#include <vector>
#include <exception>
#include "pitts_fixed_tensor3.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! tensor train class with compile-time fixed dimension
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  compile-time dimension of all ranks
  //!
  template<typename T, int N>
  class FixedTensorTrain
  {
    public:
      //! allow const access to all sub-tensors
      const auto& subTensors() const {return subTensors_;}

      //! allow non-const access to all sub-tensors
      //!
      //! \warning Do not modify sub-tensor dimensions here, only their values!
      //!
      auto& editableSubTensors() {return subTensors_;}

      //! create a new tensor train that represents a N^d tensor
      FixedTensorTrain(int d, int initial_TTrank = 1)
      {
        // create all sub-tensors
        subTensors_.resize(d);
        setTTranks(initial_TTrank);
      }

      //! number of dimensions
      inline auto nDims() const {return subTensors_.size();}

      //! size in each dimension
      static constexpr auto n() {return N;}

      //! set sub-tensor dimensions (TT-ranks), destroying all existing data
      void setTTranks(const std::vector<int>& tt_ranks)
      {
        if( tt_ranks.size() != subTensors_.size() - 1 )
          throw std::invalid_argument("TensorTrain: wrong number of TTranks!");
        // set first and last individually
        subTensors_.front().resize(1, tt_ranks.front());
        for(int i = 1; i < subTensors_.size()-1; i++)
          subTensors_[i].resize(tt_ranks[i-1], tt_ranks[i]);
        subTensors_.back().resize(tt_ranks.back(), 1);
      }

      //! set sub-tensor dimensions (TT-ranks), destroying all existing data
      void setTTranks(int tt_rank)
      {
        subTensors_.front().resize(1, tt_rank);
        for(int i = 1; i < subTensors_.size()-1; i++)
          subTensors_[i].resize(tt_rank, tt_rank);
        subTensors_.back().resize(tt_rank, 1);
      }

      //! get current sub-tensor dimensions (TT-ranks)
      std::vector<int> getTTranks() const
      {
        std::vector<int> tt_ranks(subTensors_.size() - 1);
        for(int i = 0; i < subTensors_.size() - 1; i++)
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
        if( index.size() != subTensors_.size() )
          throw std::invalid_argument("TensorTrain: multidimensional index array has wrong dimension!");
        // we use a rank of one...
        setTTranks(1);
        for(int i = 0; i < subTensors_.size(); i++)
          subTensors_[i].setUnit(0, index[i], 0);
      }

    private:
      //! actual data: array of sub-tensors
      //!
      //! resulting tensor train is defined as the tensor network:
      //!
      //! o -- o -- ... -- o -- o
      //! |    |           |    |
      //!
      //! where each "o" denotes a tensor of rank 3, respectively rank 2 on the boundaries
      //!
      std::vector<FixedTensor3<T,N>> subTensors_;
  };
}


#endif // PITTS_FIXED_TENSORTRAIN_HPP
