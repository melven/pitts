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

  //! small helper function to sum up all elements of a chunk ( return a_1 + ... + a_n )
  template<typename T>
  inline T sum(const Chunk<T>& a);

  //! small helper function to scale and sum up all elements of a chunk ( return scale * (a_1 + ... + a_n) )
  template<typename T>
  inline T scaled_sum(T scale, const Chunk<T>& a);
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
