/*! @file pitts_tensor2.hpp
* @brief Single tensor of rank 3 with dynamic dimensions
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2019-10-08
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// just import the module if we are in module mode and this file is not included from pitts_tensor2.cppm
#if defined(PITTS_USE_MODULES) && !defined(EXPORT_PITTS_TENSOR2)
import pitts_tensor2;
#define PITTS_TENSOR2_HPP
#endif

// include guard
#ifndef PITTS_TENSOR2_HPP
#define PITTS_TENSOR2_HPP

// global module fragment
#ifdef PITTS_USE_MODULES
module;
#endif

// includes
#include <vector>
#include <memory>
#include "pitts_chunk.hpp"
#include "pitts_performance.hpp"

// module export
#ifdef PITTS_USE_MODULES
export module pitts_tensor2;
# define PITTS_MODULE_EXPORT export
#endif


//! namespace for the library PITTS (parallel iterative tensor train solvers)
PITTS_MODULE_EXPORT namespace PITTS
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
      if( r1 == r1_ && r2 == r2_ )
        return;
      const auto timer = PITTS::timing::createScopedTimer<Tensor2<T>>();

      const auto n = r1*r2;
      const auto requiredChunks = std::max((long long)1, (n-1)/chunkSize+1);
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
    inline const T& operator()(long long i, long long j) const
    {
        const auto k = i + j*r1_;
        //return data_[k/chunkSize][k%chunkSize];
        const auto pdata = std::assume_aligned<ALIGNMENT>(&data_[0][0]);
        return pdata[k];
    }

    //! access matrix entries (column-wise ordering, write access through reference)
    inline T& operator()(long long i, long long j)
    {
        const auto k = i + j*r1_;
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
    static constexpr long long chunkSize = Chunk<T>::size;

  private:
    //! size of the buffer
    long long reservedChunks_ = 0;

    //! first dimension
    long long r1_ = 0;

    //! second dimension
    long long r2_ = 0;

    //! the actual data...
    std::unique_ptr<Chunk<T>[]> data_ = nullptr;
  };

  //! explicitly copy a Tensor2 object
  template<typename T>
  void copy(const Tensor2<T>& a, Tensor2<T>& b)
  {
    const auto r1 = a.r1();
    const auto r2 = a.r2();

    const auto timer = PITTS::performance::createScopedTimer<Tensor2<T>>(
        {{"r1", "r2"}, {r1, r2}},   // arguments
        {{r1*r2*kernel_info::NoOp<T>()},    // flops
         {r1*r2*kernel_info::Store<T>()+r1*r2*kernel_info::Load<T>()}}  // data
        );


    b.resize(r1, r2);

#pragma omp parallel for collapse(2) schedule(static) if(r1*r2 > 500)
    for(long long j = 0; j < r2; j++)
      for(long long i = 0; i < r1; i++)
        b(i,j) = a(i,j);
  }

  // explicit template instantiations
  template class Tensor2<float>;
  template class Tensor2<double>;
}


#endif // PITTS_TENSOR2_HPP
