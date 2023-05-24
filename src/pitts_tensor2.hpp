/*! @file pitts_tensor2.hpp
* @brief Single tensor of rank 3 with dynamic dimensions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR2_HPP
#define PITTS_TENSOR2_HPP

// includes
#include <memory>
#include "pitts_chunk.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! Const (non-mutable) view of a 2d matrix (does not own the underlying memory)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  class ConstTensor2View
  {
  public:
    //! create a tensor2 view
    //!
    //! @warning intended for internal use
    //!
    ConstTensor2View(Chunk<T> *data, long long r1, long long r2)
    {
      assert(data != nullptr);
      r1_ = r1;
      r2_ = r2;
      data_ = data;
    }

    //! create a tensor2 view (of nothing) with dimensions (0,0)
    ConstTensor2View() = default;

    //! access matrix entries (column-wise ordering, const variant)
    [[nodiscard]] inline const T& operator()(long long i, long long j) const
    {
      const auto k = i + j*r1_;
      //return data_[k/Chunk<T>::size][k%Chunk<T>::size];
      const auto pdata = std::assume_aligned<ALIGNMENT>(&data_[0][0]);
      return pdata[k];
    }

    //! first dimension 
    [[nodiscard]] inline long long r1() const {return r1_;}

    //! second dimension 
    [[nodiscard]] inline long long r2() const {return r2_;}

  protected:
    //! first dimension
    long long r1_ = 0;

    //! second dimension
    long long r2_ = 0;

    //! the actual data...
    Chunk<T> *data_ = nullptr;
  };

  //! Non-const (mutable) view of a 2d matrix (does not own the underlying memory)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  class Tensor2View : public ConstTensor2View<T>
  {
  public:
    // use base class constructors (for internal use!)
    using ConstTensor2View<T>::ConstTensor2View;

    // use base class const access
    using ConstTensor2View<T>::operator();

    //! access matrix entries (column-wise ordering, write access through reference)
    [[nodiscard]] inline T& operator()(long long i, long long j)
    {
      const auto k = i + j*this->r1_;
      //return data_[k/Chunk<T>::size][k%Chunk<T>::size];
      auto pdata = std::assume_aligned<ALIGNMENT>(&this->data_[0][0]);
      return pdata[k];
    }
  };

  //! "small" rank-2 tensor (matrix, intended to be used in a tensor train)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  class Tensor2 : public Tensor2View<T>
  {
  public:
    //! construct a new tensor with the given dimensions
    //!
    //! As a tensor network, this is:
    //!
    //!   --r1-- o --r2--
    //!
    //! @param r1   dimension of the first index, can be small
    //! @param r2   dimension of the third index, can be small
    //!
    Tensor2(long long r1, long long r2)
    {
      resize(r1,r2);
    }

    //! create a tensor with dimensions (0,0)
    //!
    //! Call resize, to do something useful with it...
    //!
    Tensor2() = default;

    //! adjust the desired tensor dimensions (destroying all data!)
    void resize(long long r1, long long r2)
    {
      // fast return without timer!
      if( r1 == this->r1_ && r2 == this->r2_ )
        return;
      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      const auto n = r1*r2;
      const auto requiredChunks = std::max((long long)1, (n-1)/Chunk<T>::size+1);
      if( requiredChunks > this->reservedChunks_ )
      {
        dataptr_.reset(new Chunk<T>[requiredChunks]);
        this->data_ = dataptr_.get();
        reservedChunks_ = requiredChunks;
      }
      this->r1_ = r1;
      this->r2_ = r2;
      // ensure padding is zero
      dataptr_[requiredChunks-1] = Chunk<T>{};
    }
  
  private:
    //! size of the buffer
    long long reservedChunks_ = 0;

    //! pointer to data with ownership
    std::unique_ptr<Chunk<T>[]> dataptr_ = nullptr;
  };

  //! explicitly copy a Tensor2 object
  template<typename T>
  void copy(const Tensor2<T>& a, Tensor2<T>& b);
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensor2_impl.hpp"
#endif

#endif // PITTS_TENSOR2_HPP
