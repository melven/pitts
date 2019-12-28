/*! @file pitts_tensor3.hpp
* @brief Single tensor of rank 3 where 2 dimensions are dynamic and one is fixed at compile-time
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-12-28
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_FIXED_TENSOR3_HPP
#define PITTS_FIXED_TENSOR3_HPP

// includes
#include <vector>
#include <memory>
#include "pitts_chunk.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! rank-3 tensor (intended to be used in a tensor train) with compile-time dimensions
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //! @tparam N  dimension of the second index
  //!
  template<typename T, int N>
  class FixedTensor3
  {
  public:
    //! create a tensor with the given dimensions
    //!
    //! As a tensor network, this is:
    //!
    //!   --r1-- o --r2--
    //!          |
    //!          N
    //!          |
    //!
    //! where N is given as a template parameter
    //!
    //! @param r1   dimension of the first index
    //! @param r2   dimension of the third index
    //!
    FixedTensor3(int r1, int r2)
    {
      resize(r1,r2);
    }

    //! create a tensor with dimensions (0,N,0)
    //!
    //! Call resize, to do something useful with it...
    //!
    FixedTensor3() = default;

    //! adjust the desired tensor dimensions (destroying all data!)
    void resize(int r1, int r2)
    {
      const auto size = r1*r2*n_;
      data_.resize(std::max(1, (size-1)/chunkSize+1));
      r1_ = r1;
      r2_ = r2;
      // ensure padding is zero
      data_.back() = Chunk<T>{};
    }

    //! access tensor entries (some block ordering, const variant)
    inline const T& operator()(int i1, int j, int i2) const
    {
      int k = i1 + j*r1_ + i2*r1_*n_;
      //return data_[k/chunkSize][k%chunkSize];
      const auto pdata = std::assume_aligned<ALIGNMENT>(&data_[0][0]);
      return pdata[k];
    }

    //! access tensor entries (some block ordering, write access through reference)
    inline T& operator()(int i1, int j, int i2)
    {
      int k = i1 + j*r1_ + i2*r1_*n_;
      //return data_[k/chunkSize][k%chunkSize];
      const auto pdata = std::assume_aligned<ALIGNMENT>(&data_[0][0]);
      return pdata[k];
    }

    //! first dimension
    inline auto r1() const {return r1_;}

    //! second dimension
    inline auto n() const {return n_;}

    //! third dimension
    inline auto r2() const {return r2_;}

    //! set all entries to the same value
    void setConstant(T v)
    {
      for(int k = 0; k < r2_; k++)
        for(int j = 0; j < n_; j++)
          for(int i = 0; i < r1_; i++)
            (*this)(i,j,k) = v;
    }

    //! set to canonical unit tensor e_(i,j,k)
    void setUnit(int ii, int jj, int kk)
    {
      for(int k = 0; k < r2_; k++)
        for(int j = 0; j < n_; j++)
          for(int i = 0; i < r1_; i++)
            (*this)(i,j,k) = (i==ii && j==jj && k==kk) ? T(1) : T(0);
    }

  protected:
    //! the array dimension of chunks
    //!
    //! (workaround for missing static function size() of std::array!)
    //!
    static constexpr int chunkSize = Chunk<T>::size;

  private:
    //! first dimension
    int r1_ = 0;

    //! second dimension
    static constexpr int n_ = N;

    //! third dimension
    int r2_ = 0;

    //! the actual data...
    std::vector<Chunk<T>> data_;
  };
}


#endif // PITTS_FIXED_TENSOR3_HPP
