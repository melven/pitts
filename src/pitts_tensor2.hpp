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
#include <vector>
#include <memory>
#include "pitts_chunk.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! "small" rank-2 tensor (matrix, intended to be used in a tensor train)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  class Tensor2
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
    Tensor2(int r1, int r2)
    {
      resize(r1,r2);
    }

    //! create a tensor with dimensions (0,0)
    //!
    //! Call resize, to do something useful with it...
    //!
    Tensor2() = default;

    //! adjust the desired tensor dimensions (destroying all data!)
    void resize(int r1, int r2)
    {
      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      const auto n = r1*r2;
      const auto requiredChunks = std::max(1, (n-1)/chunkSize+1);
      if( requiredChunks > reservedChunks_ )
      {
        data_.reset(new Chunk<T>[requiredChunks]);
        reservedChunks_ = requiredChunks;
      }
      r1_ = r1;
      r2_ = r2;
      // ensure padding is zero
      data_[requiredChunks-1] = Chunk<T>{};
    }

    //! access matrix entries (column-wise ordering, const variant)
    inline const T& operator()(int i, int j) const
    {
        int k = i + j*r1_;
        //return data_[k/chunkSize][k%chunkSize];
        const auto pdata = std::assume_aligned<ALIGNMENT>(&data_[0][0]);
        return pdata[k];
    }

    //! access matrix entries (column-wise ordering, write access through reference)
    inline T& operator()(int i, int j)
    {
        int k = i + j*r1_;
        //return data_[k/chunkSize][k%chunkSize];
        auto pdata = std::assume_aligned<ALIGNMENT>(&data_[0][0]);
        return pdata[k];
    }

    //! first dimension 
    inline auto r1() const {return r1_;}

    //! second dimension 
    inline auto r2() const {return r2_;}

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
    int r2_ = 0;

    //! the actual data...
    std::unique_ptr<Chunk<T>[]> data_ = nullptr;
  };

  //! explicitly copy a Tensor2 object
  template<typename T>
  auto copy(const Tensor2<T>& a)
  {
    const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

    const auto r1 = a.r1();
    const auto r2 = a.r2();

    Tensor2<T> b(r1, r2);

#pragma omp parallel for collapse(2) schedule(static) if(r1*r2 > 500)
    for(int j = 0; j < r2; j++)
      for(int i = 0; i < r1; i++)
        b(i,j) = a(i,j);

    return b;
  }
}


#endif // PITTS_TENSOR2_HPP
