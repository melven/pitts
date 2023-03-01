/*! @file pitts_chunk_ops_plain.hpp
* @brief default (scalar) implementation of common operations with PITTS::Chunk
* @author Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
* @date 2020-07-11
* @copyright Deutsches Zentrum fuer Luft- und Raumfahrt e. V. (DLR), German Aerospace Center
*
**/

// include guard
#ifndef PITTS_CHUNK_OPS_PLAIN_HPP
#define PITTS_CHUNK_OPS_PLAIN_HPP

// includes
#include "pitts_chunk.hpp"


//! namespace for the library PITTS (parallel iterative tensor train solvers)
namespace PITTS
{
  // Chunk FMA4 default implementation
  template<typename T>
  inline void fmadd(const Chunk<T>& a, const Chunk<T>& b, const Chunk<T>& c, Chunk<T>& d)
  {
    for(short i = 0; i < Chunk<T>::size; i++)
      d[i] = a[i]*b[i] + c[i];
  }

  // Chunk FMA3 default implementation
  template<typename T>
  inline void fmadd(const Chunk<T>& a, const Chunk<T>& b, Chunk<T>& c)
  {
    fmadd(a, b, c, c);
  }

  // scalar+Chunk FMA3 default implementation
  template<typename T>
  inline void fmadd(T a, const Chunk<T>& b, Chunk<T>& c)
  {
    for(short i = 0; i < Chunk<T>::size; i++)
      c[i] += a*b[i];
  }

  // scalare+Chunk MUL default implementation
  template<typename T>
  inline void mul(T a, const Chunk<T>& b, Chunk<T>& c)
  {
    for(short i = 0; i < Chunk<T>::size; i++)
      c[i] = a*b[i];
  }

  // Chunk MUL default implementation
  template<typename T>
  inline void mul(const Chunk<T>& a, const Chunk<T>& b, Chunk<T>& c)
  {
    for(short i = 0; i < Chunk<T>::size; i++)
      c[i] = a[i]*b[i];
  }

  // negative FMA4 default implementation
  template<typename T>
  inline void fnmadd(const Chunk<T>& a, const Chunk<T>& b, const Chunk<T>& c, Chunk<T>& d)
  {
    for(short i = 0; i < Chunk<T>::size; i++)
      d[i] = c[i] - a[i]*b[i];
  }

  // negative FMA3 default implementation
  template<typename T>
  inline void fnmadd(const Chunk<T>& a, const Chunk<T>& b, Chunk<T>& c)
  {
    fnmadd(a, b, c, c);
  }

  // horizontal add default implementation
  template<typename T>
  inline T sum(const Chunk<T>& a)
  {
    T tmp = T(0);
    for(short i = 0; i < Chunk<T>::size; i++)
      tmp += a[i];
    return tmp;
  }

  // scaled horizontal sum default implementation
  template<typename T>
  inline T scaled_sum(T scale, const Chunk<T>& a)
  {
    T tmp = T(0);
    for(short i = 0; i < Chunk<T>::size; i++)
      tmp += scale * a[i];
    return tmp;
  }

  // horizontal add + broadcast default implementation
  template<typename T>
  inline void bcast_sum(Chunk<T>& v)
  {
    T tmp = sum(v);
    for(short i = 0; i < Chunk<T>::size; i++)
      v[i] = tmp;
  }

  // masked broadcast
  template<typename T>
  inline void index_bcast(const Chunk<T>& src, short index, T value, Chunk<T>& result)
  {
    for(short i = 0; i < Chunk<T>::size; i++)
      result[i] = (i == index) ? value : src[i];
  }

  // masked load
  template<typename T>
  inline void masked_load_after(const Chunk<T>& src, short index, Chunk<T>& result)
  {
    for(short i = 0; i < Chunk<T>::size; i++)
      result[i] = (i < index) ? T(0) : src[i];
  }

  // masked store
  template<typename T>
  inline void masked_store_after(const Chunk<T>& src, short index, Chunk<T>& result)
  {
    for(short i = index; i < Chunk<T>::size; i++)
      result[i] = src[i];
  }

  // unaligned load
  template<typename T>
  inline void unaligned_load(const T* src, Chunk<T>& result)
  {
    for(short i = 0; i < Chunk<T>::size; i++)
      result[i] = src[i];
  }

  // unaligned store
  template<typename T>
  inline void unaligned_store(const Chunk<T>& src, T* result)
  {
    for(short i = 0; i < Chunk<T>::size; i++)
      result[i] = src[i];
  }

  // streaming store
  template<typename T>
  inline void streaming_store(const Chunk<T>& src, Chunk<T>& result)
  {
    result = src;
  }
}


#endif // PITTS_CHUNK_OPS_PLAIN_HPP
