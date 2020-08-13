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
#include "pitts_chunk_ops.hpp"
#include "pitts_timer.hpp"
#include "pitts_performance.hpp"

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
    MultiVector(int rows, int cols)
    {
      resize(rows,cols);
    }

    //! create a multivector with dimensions (0,0)
    //!
    //! Call resize, to do something useful with it...
    //!
    MultiVector() = default;

    //! adjust the desired multivector dimensions (destroying all data!)
    void resize(int rows, int cols)
    {
      // fast return without timer!
      if( rows == rows_ && cols == cols_ )
        return;
      const auto timer = PITTS::timing::createScopedTimer<MultiVector<T>>();

      const auto newRowChunks = (rows-1)/chunkSize+1;
      if( newRowChunks*cols > reservedChunks_ )
      {
        data_.reset(new Chunk<T>[newRowChunks*cols]);
        reservedChunks_ = newRowChunks*cols;
      }
      rows_ = rows;
      cols_ = cols;
      // ensure padding is zero
      for(int j = 0; j < cols; j++)
        chunk(newRowChunks-1, j) = Chunk<T>{};
    }

    //! access matrix entries (column-wise ordering, const variant)
    inline const T& operator()(int i, int j) const
    {
      return chunk(i/chunkSize, j)[i%chunkSize];
    }

    //! access matrix entries (column-wise ordering, write access through reference)
    inline T& operator()(int i, int j)
    {
      return chunk(i/chunkSize, j)[i%chunkSize];
    }

    //! chunk-wise access (const variant)
    inline const Chunk<T>& chunk(int i, int j) const
    {
      const auto rowChunks = (rows_-1)/chunkSize+1;
      return data_[i+j*rowChunks];
    }

    //! chunk-wise access (const variant)
    inline Chunk<T>& chunk(int i, int j)
    {
      const auto rowChunks = (rows_-1)/chunkSize+1;
      return data_[i+j*rowChunks];
    }

    //! first dimension 
    inline auto rows() const {return rows_;}

    //! second dimension 
    inline auto cols() const {return cols_;}

    //! number of chunks in first dimension
    inline auto rowChunks() const {return (rows_-1)/chunkSize+1;}

    //! total size of the data array including padding (e.g. for MPI communication)
    inline auto totalPaddedSize() const {return rowChunks()*chunkSize*cols_;}

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
    int rows_ = 0;

    //! second dimension
    int cols_ = 0;

    //! the actual data...
    std::unique_ptr<Chunk<T>[]> data_ = nullptr;
  };


  //! explicitly copy a MultiVector object
  template<typename T>
  void copy(const MultiVector<T>& a, MultiVector<T>& b)
  {
    const auto rows = a.rows();
    const auto rowChunks = a.rowChunks();
    const auto cols = a.cols();

    const auto timer = PITTS::performance::createScopedTimer<MultiVector<T>>(
        {{"ros", "cols"}, {rows, cols}},   // arguments
        {{double(rows)*cols*kernel_info::NoOp<T>()},    // flops
         {double(rows)*cols*kernel_info::Store<T>() + double(rows)*cols*kernel_info::Load<T>()}}  // data
        );

    b.resize(rows, cols);

#pragma omp parallel
    {
      for(int j = 0; j < cols; j++)
      {
#pragma omp for schedule(static) nowait
        for(int iChunk = 0; iChunk < rowChunks; iChunk++)
          streaming_store(a.chunk(iChunk,j), b.chunk(iChunk,j));
      }
    }
  }
}


#endif // PITTS_MULTIVECTOR_HPP
