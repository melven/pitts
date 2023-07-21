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
#include <memory>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include "pitts_tensor3.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! namespace for helper functionality
  namespace internal
  {
    template<typename T>
    std::vector<int> dimensionsFromSubTensors(const std::vector<Tensor3<T>>& subTensors)
    {
      std::vector<int> dims(subTensors.size());
      for(int i = 0; i < dims.size(); i++)
        dims[i] = subTensors[i].n();
      return dims;
    }
  }

  //! A tensor-train can be left- or right-orthogonal (or none of both)
  enum struct TT_Orthogonality : unsigned char
  {
    //! Currently not orthogonal or unknown
    none = 0,

    //! left-orthogonal (for all subtensors X_i: fold_left(X_i)^T fold_left(X_i) = I)
    left = 1,

    //! right-orthogonal (for all subtensors X_i: fold_right(X_i)^T fold_right(X_i) = I)
    right = 2,

    //! both left- and right-orthogonal (e.g. a unit tensor)
    both = 3
  };

  //! allow to perform bitwise-operation & on TT_Orthogonality
  constexpr TT_Orthogonality operator&(TT_Orthogonality a, TT_Orthogonality b) noexcept
  {
    return static_cast<TT_Orthogonality>(static_cast<unsigned char>(a) & static_cast<unsigned char>(b));
  }

  //! allow to perform bitwise-operation | on TT_Orthogonality
  constexpr TT_Orthogonality operator|(TT_Orthogonality a, TT_Orthogonality b) noexcept
  {
    return static_cast<TT_Orthogonality>(static_cast<unsigned char>(a) | static_cast<unsigned char>(b));
  }


  //! tensor train class
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  class TensorTrain
  {
    public:
      //! create a new tensor train that represents a n^d tensor
      TensorTrain(int d, int n, int initial_TTrank = 1) : TensorTrain(std::vector<int>(d,n), initial_TTrank) {}

      //! create a new tensor with the given dimensions
      TensorTrain(const std::vector<int>& dimensions, int initial_TTrank = 1)
        : dimensions_(dimensions),
        orthonormal_(dimensions.size(), TT_Orthogonality::none)
      {
        // create all sub-tensors
        subTensors_.resize(dimensions.size());
        for(int i = 0; i < dimensions_.size(); i++)
        {
          const int r1 = (i == 0) ? 1 : initial_TTrank;
          const int r2 = (i+1 == dimensions_.size()) ? 1 : initial_TTrank;
          subTensors_[i].resize(r1, dimensions_[i], r2);
        }
      }

      //! explicit copy construction is ok
      TensorTrain(const TensorTrain<T>& other)
        : dimensions_(other.dimensions_),
        orthonormal_(other.orthonormal_)
      {
        // create and copy all subtensors
        subTensors_.resize(dimensions_.size());
        for(int i = 0; i < dimensions_.size(); i++)
          copy(other.subTensor(i), subTensors_[i]);
      }

      //! construct from given sub-tensors, dimensions are obtained from the sub-tensor dimensions
      TensorTrain(std::vector<Tensor3<T>>&& subTensors, const std::vector<TT_Orthogonality>& orthonormal = {}) :
        dimensions_(internal::dimensionsFromSubTensors(subTensors)),
        orthonormal_(dimensions_.size(), TT_Orthogonality::none)
      {
        subTensors_.resize(dimensions_.size());
        setSubTensors(0, std::move(subTensors), orthonormal);
      }

      //! no implicit copy assignment
      const TensorTrain<T>& operator=(const TensorTrain<T>&) = delete;

      //! move construction is ok
      TensorTrain(TensorTrain<T>&&) = default;

      //! move assignment is ok
      TensorTrain<T>& operator=(TensorTrain<T>&&) = default;


      //! tensor dimensions
      //!
      //! These are constant as one usually only changes the ranks of the individual sub-tensors in the tensor train.
      //!
      const auto& dimensions() const {return dimensions_;}

      //! get i'th sub-tensor
      const Tensor3<T>& subTensor(int i) const {return subTensors_.at(i);}

      //! set i'th sub-tensor
      //!
      //! Intentionally moves from the input argument and returns the old sub-tensor.
      //! This allows to call this method without allocating or copying data and to reuse the old memory.
      //!
      Tensor3<T> setSubTensor(int i, Tensor3<T>&& newSubTensor, TT_Orthogonality orthonormal = TT_Orthogonality::none)
      {
        if( newSubTensor.n() != dimensions_.at(i) )
          throw std::invalid_argument("Invalid subtensor dimension!");
        if( i > 0 && newSubTensor.r1() != subTensors_[i].r1() )
          throw std::invalid_argument("Invalid subtensor rank (r1)!");
        if( i+1 < dimensions_.size() && newSubTensor.r2() != subTensors_[i].r2() )
          throw std::invalid_argument("Invalid subtensor rank (r2)!");

        // similar to std::swap
        Tensor3<T> oldSubTensor(std::move(subTensors_[i]));
        subTensors_[i] = std::move(newSubTensor);
        orthonormal_[i] = orthonormal;
        return oldSubTensor;
      }

      //! set i'th sub-tensor (by applying a function to it)
      template<typename Tensor3Function>
      void editSubTensor(int i, const Tensor3Function& fun, TT_Orthogonality orthonormal = TT_Orthogonality::none)
      {
        if( i < 0 || i >= subTensors_.size() )
          throw std::invalid_argument("TensorTrain::setSubTensor<Tensor3Function>: invalid sub-tensor index!");

        // not allowed to change sub-tensor dimensions through this!
        const int r1 = subTensors_[i].r1();
        const int n = subTensors_[i].n();
        const int r2 = subTensors_[i].r2();

        fun(subTensors_[i]);

        if( i > 0 && subTensors_[i].r1() != r1 )
        {
          subTensors_[i].resize(r1, n, r2);
          throw std::invalid_argument("TensorTrain::setSubTensor<Tensor3Function>: invalid subtensor rank (r1)!");
        }
        if( subTensors_[i].n() != n )
        {
          subTensors_[i].resize(r1, n, r2);
          throw std::invalid_argument("TensorTrain::setSubTensor<Tensor3Function>: invalid subtensor dimension!");
        }
        if( i+1 < dimensions_.size() && subTensors_[i].r2() != r2 )
        {
          subTensors_[i].resize(r1, n, r2);
          throw std::invalid_argument("TensorTrain::setSubTensor<Tensor3Function>: invalid subtensor rank (2)!");
        }

        orthonormal_[i] = orthonormal;
      }

      //! set a range of sub-tensors
      //!
      //! similar to setSubTensor but replaces several sub-tensors at once allowing to change the intermediate TT ranks.
      //!
      //! Intentionally moves from the input argument and returns the old sub-tensors.
      //! This allows to call this method without allocating or copying data and to reuse the old memory.
      //!
      std::vector<Tensor3<T>> setSubTensors(int offset, std::vector<Tensor3<T>>&& newSubTensors, const std::vector<TT_Orthogonality>& orthonormal = {})
      {
        for(int i = 0; i < newSubTensors.size(); i++)
        {
          // check dimensions
          if( newSubTensors[i].n() != dimensions_.at(offset+i) )
            throw std::invalid_argument("Invalid subtensor dimension!");
          // check first TT rank
          if( i == 0 && offset > 0 )
            if( newSubTensors[i].r1() != subTensors_[offset+i].r1() )
              throw std::invalid_argument("Invalid subtensor rank (r1)!");
          // check intermediate TT ranks
          if( i+1 < newSubTensors.size() )
            if( newSubTensors[i].r2() != newSubTensors[i+1].r1() )
              throw std::invalid_argument("Invalid subtensors intermediate rank (r1!=r2)!");
          // check last TT rank
          if( i+1 == newSubTensors.size() && offset+i+1 < subTensors_.size() )
            if( newSubTensors[i].r2() != subTensors_[offset+i].r2() )
              throw std::invalid_argument("Invalid subtensor rank (r2)!");
        }
        if( orthonormal.size() > 0 && orthonormal.size() != newSubTensors.size() )
          throw std::invalid_argument("Dimension of orthonormal array must match number of sub-tensors!");

        std::vector<Tensor3<T>> tmp(std::move(newSubTensors));
        for(int i = 0; i < tmp.size(); i++)
        {
          std::swap(subTensors_[offset+i], tmp[i]);
          orthonormal_[offset+i] = orthonormal.size() > 0 ? orthonormal[i] : TT_Orthogonality::none;
        }
        return tmp;
      }

      //! set a range of sub-tensors
      //!
      //! Intentionally moves from the input argument and returns the old sub-tensors.
      //! This allows to call this method without allocating or copying data and to reuse the old memory.
      //!
      TensorTrain<T> setSubTensors(int offset, TensorTrain<T>&& other)
      {
        // careful for correct behavior even for exceptions
        other.subTensors_ = setSubTensors(offset, std::move(other.subTensors_), other.orthonormal_);

        TensorTrain<T> tmp(std::move(other));
        return tmp;
      }

      //! set sub-tensor dimensions (TT-ranks), destroying all existing data
      void setTTranks(const std::vector<int>& tt_ranks)
      {
        if( tt_ranks.size() != dimensions_.size() - 1 )
          throw std::invalid_argument("TensorTrain: wrong number of TTranks!");
        // Usually r0 = rd = 1 but...
        // there are some special cases where we use r0,rd != 1
        // ("boundary rank" which behaves like an additional dimension to avoid
        // appending a unit matrices as first and last subtensors)
        const auto r0 = subTensors_.front().r1();
        const auto rd = subTensors_.back().r2();

        for(int i = 0; i < dimensions_.size(); i++)
        {
          const int r1 = (i == 0) ? r0 : tt_ranks[i-1];
          const int r2 = (i+1 == dimensions_.size()) ? rd : tt_ranks[i];
          subTensors_[i].resize(r1, dimensions_[i], r2);
        }
      }

      //! set sub-tensor dimensions (TT-ranks), destroying all existing data
      void setTTranks(int tt_rank)
      {
        // preserve boundary ranks (see above)
        const auto r0 = subTensors_.front().r1();
        const auto rd = subTensors_.back().r2();

        for(int i = 0; i < dimensions_.size(); i++)
        {
          const int r1 = (i == 0) ? r0 : tt_rank;
          const int r2 = (i+1 == dimensions_.size()) ? rd : tt_rank;
          subTensors_[i].resize(r1, dimensions_[i], r2);
        }
      }

      //! get current sub-tensor dimensions (TT-ranks)
      std::vector<int> getTTranks() const
      {
        std::vector<int> tt_ranks(dimensions_.size() - 1);
        for(int i = 0; i < dimensions_.size() - 1; i++)
          tt_ranks[i] = subTensors_[i].r2();
        return tt_ranks;
      }

      //! current orthognoality state of the i-th sub-tensor
      [[nodiscard]] TT_Orthogonality isOrthonormal(int i) const {return orthonormal_.at(i);}

      //! current orthogonality state of the tensor-train
      //!
      //! Calculated from the orthogonality state of the sub-tensors
      //!
      [[nodiscard]] TT_Orthogonality isOrthogonal() const noexcept
      {
        const bool leftOrtho = std::all_of(orthonormal_.begin(), orthonormal_.end()-1, [](TT_Orthogonality o) -> bool {return (o & TT_Orthogonality::left) != TT_Orthogonality::none;});
        const bool rightOrtho = std::all_of(orthonormal_.begin()+1, orthonormal_.end(), [](TT_Orthogonality o) -> bool {return (o & TT_Orthogonality::right) != TT_Orthogonality::none;});
        TT_Orthogonality result = TT_Orthogonality::none;
        if( leftOrtho )
          result = result | TT_Orthogonality::left;
        if( rightOrtho )
          result = result | TT_Orthogonality::right;
        return result;
      }

      //! current orthonormality state of the tensor-train (orthogonal + norm 1)
      //!
      //! Calculated from the orthogonality state of the sub-tensors
      //!
      [[nodiscard]] TT_Orthogonality isOrthonormal() const noexcept
      {
        const bool leftOrtho = std::all_of(orthonormal_.begin(), orthonormal_.end(), [](TT_Orthogonality o) {return (o & TT_Orthogonality::left) != TT_Orthogonality::none;});
        const bool rightOrtho = std::all_of(orthonormal_.begin(), orthonormal_.end(), [](TT_Orthogonality o) {return (o & TT_Orthogonality::right) != TT_Orthogonality::none;});
        TT_Orthogonality result = TT_Orthogonality::none;
        if( leftOrtho )
          result = result | TT_Orthogonality::left;
        if( rightOrtho )
          result = result | TT_Orthogonality::right;
        return result;
      }

      //! make this a tensor of zeros
      //!
      //! \warning intended for testing purposes
      //!
      void setZero()
      {
        // we use a rank of one...
        setTTranks(1);
        for(int i = 0; i < subTensors_.size(); i++)
        {
          subTensors_[i].setConstant(T(0));
          orthonormal_[i] = TT_Orthogonality::none;
        }
      }

      //! make this a tensor of ones
      //!
      //! \warning intended for testing purposes
      //!
      void setOnes()
      {
        // we use a rank of one...
        setTTranks(1);
        for(int i = 0; i < subTensors_.size(); i++)
        {
          subTensors_[i].setConstant(T(1));
          orthonormal_[i] = TT_Orthogonality::none;
        }
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
        {
          subTensors_[i].setUnit(0, index[i], 0);
          orthonormal_[i] = TT_Orthogonality::both;
        }
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

      //! current orthogonality state of the sub-tensors of the tensor-train
      //!
      //! This is set by e.g. setSubTensor and setSubTensors.
      //!
      std::vector<TT_Orthogonality> orthonormal_;
  };

  //! explicitly copy a TensorTrain object
  template<typename T>
  void copy(const TensorTrain<T>& a, TensorTrain<T>& b);
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensortrain_impl.hpp"
#endif


#endif // PITTS_TENSORTRAIN_HPP
