/*! @file pitts_multivector.hpp
* @brief Rank-2 tensor that represents a set of large vectors (e.g. leading dimension is large)
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-02-09
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_MULTIVECTOR_HPP
#define PITTS_MULTIVECTOR_HPP

// includes
#include <memory>
#include "pitts_chunk.hpp"
#include "pitts_timer.hpp"

//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! "Set" of large vectors (matrix with a high number of rows)
  //!
  //! @tparam T  underlying data type (double, complex, ...)
  //!
  template<typename T>
  class MultiVector
  {
  public:
    //! construct a new multivector with the given dimensions
    //!
    //! As a tensor network, this is:
    //!
    //!   --rows-- o --cols--
    //!
    //! @param rows   dimension of the first index, should be large
    //! @param cols   dimension of the third index, can be small
    //!
    MultiVector(long long rows, long long cols)
    {
      resize(rows,cols);
    }

    //! create a multivector with dimensions (0,0)
    //!
    //! Call resize, to do something useful with it...
    //!
    MultiVector() = default;

    //! adjust the desired multivector dimensions (destroying all data!)
    void resize(long long rows, long long cols)
    {
      // fast return without timer!
      if( rows == rows_ && cols <= cols_ )
      {
        cols_ = cols;
        return;
      }
      const auto timer = PITTS::timing::createScopedTimer<MultiVector<T>>();

      const auto newRowChunks = (rows-1)/chunkSize+1;
      const auto newColStrideChunks = internal::paddedChunks(newRowChunks);
      if( newColStrideChunks*cols > reservedChunks_ )
      {
        data_.reset(new Chunk<T>[newColStrideChunks*cols]);
        reservedChunks_ = newColStrideChunks*cols;
      }
      rows_ = rows;
      cols_ = cols;
      // ensure padding is zero
      for(long long j = 0; j < cols; j++)
        chunk(newRowChunks-1, j) = Chunk<T>{};
    }

    //! access matrix entries (column-wise ordering, const variant)
    inline const T& operator()(long long i, long long j) const
    {
      return chunk(i/chunkSize, j)[i%chunkSize];
    }

    //! access matrix entries (column-wise ordering, write access through reference)
    inline T& operator()(long long i, long long j)
    {
      return chunk(i/chunkSize, j)[i%chunkSize];
    }

    //! chunk-wise access (const variant)
    inline const Chunk<T>& chunk(long long i, long long j) const
    {
      return data_[i+j*colStrideChunks()];
    }

    //! chunk-wise access (const variant)
    inline Chunk<T>& chunk(long long i, long long j)
    {
      return data_[i+j*colStrideChunks()];
    }

    //! first dimension 
    inline auto rows() const {return rows_;}

    //! second dimension 
    inline auto cols() const {return cols_;}

    //! total size
    inline auto size() const {return rows_*cols_;}

    //! number of chunks in first dimension
    inline auto rowChunks() const {return (rows_-1)/chunkSize+1;}

    //! number of chunks between two columns (stride in second dimension)
    inline auto colStrideChunks() const
    {
      // return uneven number to avoid cache thrashing
      return internal::paddedChunks(rowChunks());
    }

    //! total size of the data array including padding (e.g. for MPI communication)
    inline auto totalPaddedSize() const {return colStrideChunks()*chunkSize*cols_;}

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
    long long rows_ = 0;

    //! second dimension
    long long cols_ = 0;

    //! the actual data...
    std::unique_ptr<Chunk<T>[]> data_ = nullptr;
  };


  //! explicitly copy a MultiVector object
  template<typename T>
  void copy(const MultiVector<T>& a, MultiVector<T>& b);
  
}

#ifndef PITTS_DEVELOP_BUILD
#include "pitts_multivector_impl.hpp"
#endif

#endif // PITTS_MULTIVECTOR_HPP
