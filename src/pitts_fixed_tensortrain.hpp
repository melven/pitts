// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_fixed_tensortrain.hpp
* @brief simple tensor train format with compile-time dimension
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-28
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

      //! explicit copy construction is ok
      FixedTensorTrain(const FixedTensorTrain<T,N>& other)
      {
        // create and copy all subtensors
        subTensors_.resize(other.nDims());
        for(int i = 0; i < other.nDims(); i++)
          copy(other.subTensors()[i], subTensors_[i]);
      }

      //! no implicit copy assignment
      const FixedTensorTrain<T,N>& operator=(const FixedTensorTrain<T,N>&) = delete;

      //! move construction is ok
      FixedTensorTrain(FixedTensorTrain<T,N>&&) = default;

      //! move assignment is ok
      FixedTensorTrain<T,N>& operator=(FixedTensorTrain<T,N>&&) = default;

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
        for(int i = 0; i < nDims(); i++)
        {
          const auto r1 = i > 0 ? tt_rank : 1;
          const auto r2 = i+1 < nDims() ? tt_rank : 1;
          subTensors_[i].resize(r1, r2);
        }
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

  //! explicitly copy a TensorTrain object
  template<typename T, int N>
  void copy(const FixedTensorTrain<T,N>& a, FixedTensorTrain<T,N>& b)
  {
    // check that dimensions match
    if( a.nDims() != b.nDims() )
      throw std::invalid_argument("FixedTensorTrain copy dimension mismatch!");

    for(int i = 0; i < a.nDims(); i++)
      copy(a.subTensors()[i], b.editableSubTensors()[i]);
  }
}


#endif // PITTS_FIXED_TENSORTRAIN_HPP
