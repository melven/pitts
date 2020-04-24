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
    Tensor3(int r1, int n, int r2)
    {
      resize(r1,n,r2);
    }

    //! create a tensor with dimensions (0,0,0)
    //!
    //! Call resize, to do something useful with it...
    //!
    Tensor3() = default;

    //! adjust the desired tensor dimensions (destroying all data!)
    void resize(int r1, int n, int r2)
    {
      const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

      const auto requiredChunks = r1 * r2 * std::max(1, (n-1)/chunkSize+1);
      if( requiredChunks > reservedChunks_ )
      {
        data_.reset(new Chunk<T>[requiredChunks]);
        reservedChunks_ = requiredChunks;
      }
      r1_ = r1;
      r2_ = r2;
      n_ = n;
      // ensure padding is zero
      for(int j = 0; j < r2_; j++)
        for (int i = 0; i < r1_; i++)
          chunk(i,nChunks()-1,j) = Chunk<T>{};
    }

    //! access tensor entries (some block ordering, const variant)
    inline const T& operator()(int i1, int j, int i2) const
    {
      const int k = i1 + j/chunkSize*r1_ + i2*r1_*nChunks();
      return data_[k][j%chunkSize];
    }

    //! access tensor entries (some block ordering, write access through reference)
    inline T& operator()(int i1, int j, int i2)
    {
      const int k = i1 + j/chunkSize*r1_ + i2*r1_*nChunks();
      return data_[k][j%chunkSize];
    }

    //! chunk-wise access
    const Chunk<T>& chunk(int i1, int j, int i2) const
    {
      const int k = i1 + j*r1_ + i2*r1_*nChunks();
      const auto pdata = std::assume_aligned<ALIGNMENT>(data_.get());
      return pdata[k];
    }

    //! chunk-wise access
    Chunk<T>& chunk(int i1, int j, int i2)
    {
      const int k = i1 + j*r1_ + i2*r1_*nChunks();
      auto pdata = std::assume_aligned<ALIGNMENT>(data_.get());
      return pdata[k];
    }

    //! first dimension
    inline auto r1() const {return r1_;}

    //! second dimension
    inline auto n() const {return n_;}

    //! number  of chunks in the second dimension
    inline auto nChunks() const {return (n_-1)/chunkSize+1;}

    //! third dimension
    inline auto r2() const {return r2_;}

    //! set all entries to the same value
    void setConstant(T v)
    {
      const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

      for(int i = 0; i < r1_; i++)
        for(int j = 0; j < n_; j++)
          for(int k = 0; k < r2_; k++)
            (*this)(i,j,k) = v;
    }

    //! set to canonical unit tensor e_(i,j,k)
    void setUnit(int ii, int jj, int kk)
    {
      const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

      for(int i = 0; i < r1_; i++)
        for(int j = 0; j < n_; j++)
          for(int k = 0; k < r2_; k++)
            (*this)(i,j,k) = (i==ii && j==jj && k==kk) ? T(1) : T(0);
    }

  protected:
    //! the array dimension of chunks
    //!
    //! (workaround for missing static function size() of std::array!)
    //!
    static constexpr int chunkSize = Chunk<T>::size;

  private:
    //! size of the buffer
    int reservedChunks_ = 0;

    //! first dimension
    int r1_ = 0;

    //! second dimension
    int n_ = 0;

    //! third dimension
    int r2_ = 0;

    //! the actual data...
    std::unique_ptr<Chunk<T>[]> data_ = nullptr;
  };

  //! explicitly copy a Tensor3 object
  template<typename T>
  void copy(const Tensor3<T>& a, Tensor3<T>& b)
  {
    const auto timer = PITTS::timing::createScopedTimer<Tensor3<T>>();

    const auto r1 = a.r1();
    const auto n = a.n();
    const auto r2 = a.r2();

    b.resize(r1, n, r2);

#pragma omp parallel for collapse(3) schedule(static) if(r1*n*r2 > 500)
      for(int i = 0; i < r1; i++)
        for(int j = 0; j < n; j++)
          for(int k = 0; k < r2; k++)
            b(i,j,k) = a(i,j,k);
  }
}


#endif // PITTS_TENSOR3_HPP
