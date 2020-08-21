/*! @file pitts_chunk_ops.hpp
* @brief helper functions for common SIMD operations using PITTS::Chunk
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-06-30
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_CHUNK_OPS_HPP
#define PITTS_CHUNK_OPS_HPP

// includes
#include "pitts_chunk.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  //! add the element-wise product of two chunks ( d_i = a_i * b_i + c_i )
  template<typename T>
  inline void fmadd(const Chunk<T>& a, const Chunk<T>& b, const Chunk<T>& c, Chunk<T>& d);

  //! add the element-wise product of two chunks ( c_i += a_i * b_i )
  template<typename T>
  inline void fmadd(const Chunk<T>& a, const Chunk<T>& b, Chunk<T>& c);

  //! add the element-wise product of a scalar and a chunk ( c_i += a * b_i )
  template<typename T>
  inline void fmadd(T a, const Chunk<T>& b, Chunk<T>& c);

  //! multiply all elements in a chunk with a scalar ( c_i = a * b_i )
  template<typename T>
  inline void mul(T a, const Chunk<T>& b, Chunk<T>& c);

  //! add the negative element-wise product of two chunks ( d_i = - a_i * b_i + c_i )
  template<typename T>
  inline void fnmadd(const Chunk<T>& a, const Chunk<T>& b, const Chunk<T>& c, Chunk<T>& d);

  //! add the negative element-wise product of two chunks ( c_i -= a_i * b_i )
  template<typename T>
  inline void fnmadd(const Chunk<T>& a, const Chunk<T>& b, Chunk<T>& c);

  //! sum up all elements of a chunk ( return a_1 + ... + a_n )
  template<typename T>
  inline T sum(const Chunk<T>& a);

  //! scale and sum up all elements of a chunk ( return scale * (a_1 + ... + a_n) )
  template<typename T>
  inline T scaled_sum(T scale, const Chunk<T>& a);

  //! sum up all elements of a chunk and broadcast the result to all elements ( a_i = a_1 + ... a_n )
  template<typename T>
  inline void bcast_sum(Chunk<double>& v);

  //! masked broadcast to given index, sets result to value at given index and to src everywhere else ( result_i = (i==index) ? value : src_i )
  template<typename T>
  inline void index_bcast(const Chunk<T>& src, short index, T value, Chunk<T>& result);

  //! masked load: load all values after a given index, zero out first values ( result_i = (i < startIndex) ? 0 : src_i )
  template<typename T>
  inline void masked_load_after(const Chunk<T>& src, short index, Chunk<T>& result);

  //! masked store: store all values after a given index, keep first values ( result_i = (i < startIndex) ? result_i : src_i )
  template<typename T>
  inline void masked_store_after(const Chunk<T>& src, short index, Chunk<T>& result);

  //! unaligned load: read chunk from memory address with unknown alignment
  template<typename T>
  inline void unaligned_load(const T* src, Chunk<T>& result);

  //! unaligned store: write chunk to memory address with unknown alignment
  template<typename T>
  inline void unaligned_store(const Chunk<T>& src, T* result);

  //! streaming store: write chunk to memory using a non-temporal store hint
  template<typename T>
  inline void streaming_store(const Chunk<T>& src, Chunk<T>& result);
}


// include appropriate implementation
#if defined(__AVX512F__)
# include "pitts_chunk_ops_avx512.hpp"
#elif defined(__AVX2__)
# include "pitts_chunk_ops_avx2.hpp"
#else
# include "pitts_chunk_ops_plain.hpp"
#endif


#endif // PITTS_CHUNK_OPS_HPP
