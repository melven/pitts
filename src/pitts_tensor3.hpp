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
#include <vector>
#include "pitts_chunk.hpp"

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
      data_.resize(r1 * r2 * std::max(1, n/chunkSize));
      r1_ = r1;
      r2_ = r2;
      n_ = n;
    }

    //! access matrix entries (some block ordering, const variant)
    inline T operator()(int i1, int j, int i2) const
    {
      int k = i1 + i2*r1_ + (j/chunkSize)*r1_*r2_;
      return data_[k][j%chunkSize];
    }

    //! access matrix entries (some block ordering, write access through reference)
    inline T& operator()(int i1, int j, int i2)
    {
      int k = i1 + i2*r1_ + (j/chunkSize)*r1_*r2_;
      return data_[k][j%chunkSize];
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
      for(int i = 0; i < r1_; i++)
        for(int j = 0; j < n_; j++)
          for(int k = 0; k < r2_; k++)
            (*this)(i,j,k) = v;
    }

    //! set to canonical unit tensor e_(i,j,k)
    void setUnit(int ii, int jj, int kk)
    {
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
    //! first dimension
    int r1_ = 0;

    //! second dimension
    int n_ = 0;

    //! third dimension
    int r2_ = 0;

    //! the actual data...
    std::vector<Chunk<T>> data_;
  };
}


#endif // PITTS_TENSOR3_HPP
