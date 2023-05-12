/*! @file pitts_tensor3.hpp
* @brief Single tensor of rank 3 with dynamic dimensions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_TENSOR3_HPP
#define PITTS_TENSOR3_HPP

// includes
#include <memory>
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
      if(r1*n*r2 > reservedChunks * Chunk<T>::size)
        throw std::invalid_argument("Reserved data size too small!");
      if(nullptr == data.get())
        throw std::invalid_argument("Data pointer must be allocated!");
      
      data_ = std::move(data);
      reservedChunks_ = reservedChunks;
      resize(r1, n, r2, false);
    }

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

    //! adjust the desired tensor dimensions (destroying all data!)
    void resize(long long r1, long long n, long long r2, bool setPaddingToZero = true)
    {
      // fast return without timer!
      if( r1 == r1_ && n == n_ && r2 == r2_ )
        return;
      const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

      const long long requiredSize = r1*n*r2;
      // ensure same amount of padding as in MultiVector
      const long long requiredChunks = internal::paddedChunks((requiredSize-1)/Chunk<T>::size+1);
      if( requiredChunks > reservedChunks_ )
      {
        data_.reset(new Chunk<T>[requiredChunks]);
        reservedChunks_ = requiredChunks;
      }
      r1_ = r1;
      r2_ = r2;
      n_ = n;
      if (setPaddingToZero)
        data_[requiredChunks-1] = Chunk<T>{};
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
