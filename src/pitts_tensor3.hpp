// Copyright (c) 2019 German Aerospace Center (DLR), Institute for Software Technology, Germany
// SPDX-FileContributor: Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// SPDX-FileContributor: Manuel Joey Becklas
//
// SPDX-License-Identifier: BSD-3-Clause

/*! @file pitts_tensor3.hpp
* @brief Single tensor of rank 3 with dynamic dimensions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
*
**/

// include guard
#ifndef PITTS_TENSOR3_HPP
#define PITTS_TENSOR3_HPP

// includes
#include <memory>
#include <utility>
#include "pitts_chunk.hpp"
#include "pitts_timer.hpp"
#include "pitts_performance.hpp"
#include<cassert>

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! rank-3 tensor (intended to be used in a tensor train)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  class Tensor3
  {
  public:
    //! create a tensor with the given dimensions
    //!
    //! As a tensor network, this is:
    //!
    //!   --r1-- o --r2--
    //!          |
    //!          n
    //!          |
    //!
    //! @param r1   dimension of the first index, can be small
    //! @param n    dimension of the second index, should be large
    //! @param r2   dimension of the third index, can be small
    //!
    Tensor3(long long r1, long long n, long long r2)
    {
      resize(r1,n,r2);
    }

    //! create a tensor with dimensions (0,0,0)
    //!
    //! Call resize, to do something useful with it...
    //!
    Tensor3() = default;

    //! create a tensor from given memory with given size
    //!
    //! @warning intended for internal use (e.g. fold function)
    //!
    //! @param data           pointer to the reserved memory, must be of size reservedChunks
    //! @param reservedChunks dimension of data array, must be at least big enough for r1*n*r2 / Chunk<T>::size
    //! @param r1             dimension of the first index, can be small
    //! @param n              dimension of the second index, should be large
    //! @param r2             dimension of the third index, can be small
    //!
    Tensor3(std::unique_ptr<Chunk<T>[]>&& data, long long reservedChunks, long long r1, long long n, long long r2)
    {
      if(nullptr == data.get())
        throw std::invalid_argument("Data pointer must be allocated!");
      
      data_ = std::move(data);
      reservedChunks_ = reservedChunks;
      resize(r1, n, r2, false, true);
    }

    //! move construction operator: moved-from object should be empty
    Tensor3(Tensor3<T>&& other) noexcept
    {
      *this = std::move(other);
    }

    //! move assignmen operator: moved-from object may reuse the memory of this...
    Tensor3<T>& operator=(Tensor3<T>&& other) noexcept
    {
      // std::swap(*this, other) won't work unfortunately as swap uses move assignment internally!
      std::swap(reservedChunks_, other.reservedChunks_);
      r1_ = std::exchange(other.r1_, 0);
      n_ = std::exchange(other.n_, 0);
      r2_ = std::exchange(other.r2_, 0);
      std::swap(data_, other.data_);
      return *this;
    }

    //! allow access to the internal data pointer
    //!
    //! @warning intended for internal use
    //!
    [[nodiscard]] inline Chunk<T>* data() {return data_.get();}

    //! allow read-only access to the internal data pointer
    //!
    //! @warning intended for internal use
    //!
    [[nodiscard]] inline const Chunk<T>* data() const {return data_.get();}

    //! allow to move from this by casting to the underlying storage type
    //!
    //! @warning intended for internal use (e.g. unfold function)
    //!
    [[nodiscard]] operator std::unique_ptr<Chunk<T>[]>() &&
    {
      r1_ = n_ = r2_ = 0;
      reservedChunks_ = 0;
      std::unique_ptr<Chunk<T>[]> data = std::move(data_);
      return data;
    }

    //! adjust the desired tensor dimensions (usually destroying all data!)
    //!
    //! @param r1               first dimension
    //! @param n                second dimension
    //! @param r2               third dimension
    //! @param setPaddingToZero can be set to false to avoid initialization to zero of the last chunk in each column
    //! @param keepData         try to change the dimensions without changing the data (reshape), throws an error if not enough memory was allocated
    //!
    void resize(long long r1, long long n, long long r2, bool setPaddingToZero = true, bool keepData = false)
    {
      // fast return without timer!
      if( r1 == r1_ && n == n_ && r2 == r2_ )
        return;
      const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

      const long long requiredSize = r1*n*r2;
      const long long nChunks = (requiredSize-1)/Chunk<T>::size+1;
      // ensure same amount of padding as in MultiVector
      const long long requiredChunks = internal::paddedChunks(nChunks);
      if( requiredChunks > reservedChunks_ )
      {
        if( keepData )
          throw std::invalid_argument("MultiVector: cannot resize without allocating memory!");
        data_.reset(new Chunk<T>[requiredChunks]);
        reservedChunks_ = requiredChunks;
      }
      r1_ = r1;
      r2_ = r2;
      n_ = n;
      if (setPaddingToZero)
        data_[nChunks-1] = Chunk<T>{};
    }

    //! access tensor entries (some block ordering, const variant)
    [[nodiscard]] inline const T& operator()(long long i1, long long j, long long i2) const
    {
      assert(0 <= i1 && i1 < r1_);
      assert(0 <= j  &&  j < n_);
      assert(0 <= i2 && i2 < r2_);
      const auto index = i1 + j*r1_ + i2*r1_*n_;
      //return data_[index/chunkSize][index%chunkSize];
      const auto pdata = std::assume_aligned<ALIGNMENT>(&data_[0][0]);
      return pdata[index];
    }

    //! access tensor entries (some block ordering, write access through reference)
    [[nodiscard]] inline T& operator()(long long i1, long long j, long long i2)
    {
      assert(0 <= i1 && i1 < r1_);
      assert(0 <= j  &&  j < n_);
      assert(0 <= i2 && i2 < r2_);
      const auto index = i1 + j*r1_ + i2*r1_*n_;
      //return data_[index/chunkSize][index%chunkSize];
      const auto pdata = std::assume_aligned<ALIGNMENT>(&data_[0][0]);
      return pdata[index];
    }

    //! first dimension
    [[nodiscard]] inline long long r1() const {return r1_;}

    //! second dimension
    [[nodiscard]] inline long long n() const {return n_;}

    //! third dimension
    [[nodiscard]] inline long long r2() const {return r2_;}

    //! reserved memory (internal use)
    [[nodiscard]] inline long long reservedChunks() const {return reservedChunks_;}

    //! set all entries to the same value
    void setConstant(T v)
    {
      const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
          {{"r1", "n", "r2"}, {r1_, n_, r2_}},   // arguments
          {{r1_*n_*r2_*kernel_info::NoOp<T>()},    // flops
           {r1_*n_*r2_*kernel_info::Store<T>()}}  // data
          );

      for(long long i = 0; i < r1_; i++)
        for(long long j = 0; j < n_; j++)
          for(long long k = 0; k < r2_; k++)
            (*this)(i,j,k) = v;
    }

    //! set to canonical unit tensor e_(i,j,k)
    void setUnit(long long ii, long long jj, long long kk)
    {
      const auto timer = PITTS::performance::createScopedTimer<Tensor3<T>>(
          {{"r1", "n", "r2"}, {r1_, n_, r2_}},   // arguments
          {{r1_*n_*r2_*kernel_info::NoOp<T>()},    // flops
           {r1_*n_*r2_*kernel_info::Store<T>()}}  // data
          );

      for(long long i = 0; i < r1_; i++)
        for(long long j = 0; j < n_; j++)
          for(long long k = 0; k < r2_; k++)
            (*this)(i,j,k) = (i==ii && j==jj && k==kk) ? T(1) : T(0);
    }

  private:
    //! size of the buffer
    long long reservedChunks_ = 0;

    //! first dimension
    long long r1_ = 0;

    //! second dimension
    long long n_ = 0;

    //! third dimension
    long long r2_ = 0;

    //! the actual data...
    std::unique_ptr<Chunk<T>[]> data_ = nullptr;
  };

  //! explicitly copy a Tensor3 object
  template<typename T>
  void copy(const Tensor3<T>& a, Tensor3<T>& b);
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_tensor3_impl.hpp"
#endif

#endif // PITTS_TENSOR3_HPP
