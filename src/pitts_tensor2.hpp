// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor2.hpp
* @brief Single tensor of rank 3 with dynamic dimensions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
*
**/

// include guard
#ifndef PITTS_TENSOR2_HPP
#define PITTS_TENSOR2_HPP

// includes
#include <cassert>
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
    //! data type
    using Type = T;

    //! create a tensor2 view
    //!
    //! @warning intended for internal use
    //!
    ConstTensor2View(Chunk<T> *data, long long r1, long long r2)
    {
      assert(r1*r2 == 0 || data != nullptr);
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

    //! create a tensor from given memory with given size
    //!
    //! @warning intended for internal use (e.g. fold function)
    //!
    //! @param data           pointer to the reserved memory, must be of size reservedChunks
    //! @param reservedChunks dimension of data array, must be at least big enough for r1*n*r2 / Chunk<T>::size
    //! @param r1             dimension of the first index, can be small
    //! @param r2             dimension of the third index, can be small
    //!
    Tensor2(std::unique_ptr<Chunk<T>[]>&& data, long long reservedChunks, long long r1, long long r2)
    {
      if(nullptr == data.get())
        throw std::invalid_argument("Data pointer must be allocated!");
      
      dataptr_ = std::move(data);
      this->data_ = dataptr_.get();
      reservedChunks_ = reservedChunks;
      this->r1_ = r1;
      this->r2_ = r2;
    }

    //! move construction operator: moved-from object should be empty
    Tensor2(Tensor2<T>&& other) noexcept
    {
      *this = std::move(other);
    }

    //! move assignmen operator: moved-from object may reuse the memory of this...
    Tensor2<T>& operator=(Tensor2<T>&& other) noexcept
    {
      static_cast<Tensor2View<T>&>(*this) = other;
      static_cast<Tensor2View<T>&>(other) = Tensor2View<T>();
      std::swap(reservedChunks_, other.reservedChunks_);
      std::swap(dataptr_, other.dataptr_);
      return *this;
    }

    //! allow to move from this by casting to the underlying storage type
    //!
    //! @warning intended for internal use (e.g. unfold function)
    //!
    [[nodiscard]] operator std::unique_ptr<Chunk<T>[]>() &&
    {
      std::unique_ptr<Chunk<T>[]> data = std::move(dataptr_);
      dataptr_ = nullptr;
      this->data_ = nullptr;
      this->r1_ = this->r2_ = 0;
      reservedChunks_ = 0;
      return data;
    }

    //! adjust the desired tensor dimensions (destroying all data!)
    void resize(long long r1, long long r2, bool setPaddingToZero = true)
    {
      // fast return without timer!
      if( r1 == this->r1_ && r2 == this->r2_ )
        return;
      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      const auto n = r1*r2;
      const auto requiredChunks = internal::paddedChunks((n-1)/Chunk<T>::size+1);
      if( requiredChunks > this->reservedChunks_ )
      {
        dataptr_.reset(new Chunk<T>[requiredChunks]);
        this->data_ = dataptr_.get();
        reservedChunks_ = requiredChunks;
      }
      this->r1_ = r1;
      this->r2_ = r2;
      // ensure padding is zero
      if( setPaddingToZero )
        dataptr_[requiredChunks-1] = Chunk<T>{};
    }

    //! reserved memory (internal use)
    [[nodiscard]] inline long long reservedChunks() const {return reservedChunks_;}
  
  private:
    //! size of the buffer
    long long reservedChunks_ = 0;

    //! pointer to data with ownership
    std::unique_ptr<Chunk<T>[]> dataptr_ = nullptr;
  };

  //! explicitly copy a Tensor2 object
  template<typename T>
  void copy(const Tensor2<T>& a, Tensor2<T>& b);

  //! namespace for helper functionality
  namespace internal
  {
    //! copy a Tensor2 into an array-like object with element access b(i, j)
    //!
    //! E.g. the mutable_unchecked<2>() proxy of a numpy array in pybind11 (any memory layout). b must already have the
    //! dimensions r1 x r2 (not checked). OpenMP parallel: writing different entries of b must be thread-safe.
    //! (Defined here and not in the _impl.hpp: templated on the array type, so no explicit instantiation.)
    //!
    //! @tparam T       underlying data type (double, complex, ...)
    //! @tparam Array2  type with an element access b(i, j) = value (long long indices)
    //!
    //! @param a  the matrix to copy
    //! @param b  the destination
    //!
    template<typename T, typename Array2>
      requires requires(Array2 b, const T value) { b(0LL, 0LL) = value; }
    void copyTo(const Tensor2<T>& a, Array2&& b)
    {
      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      const long long r1 = a.r1(), r2 = a.r2();
#pragma omp parallel for schedule(static) if(r1*r2 > 100000)
      for(long long j = 0; j < r2; j++)
        for(long long i = 0; i < r1; i++)
          b(i, j) = a(i, j);
    }

    //! copy an array-like object with element access a(i, j) into a Tensor2 (the counterpart of copyTo)
    //!
    //! b must already have the dimensions of a (resize it first). OpenMP parallel: reading different entries of a must
    //! be thread-safe.
    //!
    //! @tparam T       underlying data type (double, complex, ...)
    //! @tparam Array2  type with an element access a(i, j) convertible to T (long long indices)
    //!
    //! @param a  the source
    //! @param b  the destination matrix with the dimensions of a
    //!
    template<typename T, typename Array2>
      requires requires(const Array2& a) { T(a(0LL, 0LL)); }
    void copyFrom(const Array2& a, Tensor2<T>& b)
    {
      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      const long long r1 = b.r1(), r2 = b.r2();
#pragma omp parallel for schedule(static) if(r1*r2 > 100000)
      for(long long j = 0; j < r2; j++)
        for(long long i = 0; i < r1; i++)
          b(i, j) = a(i, j);
    }
  }
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensor2_impl.hpp"
#endif

#endif // PITTS_TENSOR2_HPP
